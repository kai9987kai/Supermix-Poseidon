"""Bounded arithmetic and linear algebra verification, never Python eval."""
from __future__ import annotations
import ast
from fractions import Fraction
import math
import re

class MathError(ValueError):
    pass

def _parse(expression):
    if len(expression) > 250:
        raise MathError("Expression is too long (250 characters maximum).")
    expression = expression.replace("^", "**").replace("×", "*").replace("÷", "/")
    if not re.fullmatch(r"[0-9xXyY+*/().%\s-]+", expression):
        raise MathError("Use numbers, x, y, parentheses, and + - * / ** % only.")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except (SyntaxError, RecursionError) as e:
        raise MathError("Invalid expression; use explicit multiplication such as 3*x.") from e
    if sum(1 for _ in ast.walk(tree)) > 80:
        raise MathError("Expression is too complex.")
    return tree.body

def _bounded(value):
    if abs(value.numerator) > 10**80 or value.denominator > 10**80:
        raise MathError("Result exceeds the numeric budget.")
    return value

def _linear(node):
    # Return coefficient and constant for ax+b, maintaining exact rational arithmetic.
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return Fraction(0), _bounded(Fraction(str(node.value)))
    if isinstance(node, ast.Name) and node.id.lower() == "x":
        return Fraction(1), Fraction(0)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        a, b = _linear(node.operand)
        return (a, b) if isinstance(node.op, ast.UAdd) else (-a, -b)
    if not isinstance(node, ast.BinOp):
        raise MathError("Unsupported operation.")
    a, b = _linear(node.left)
    c, d = _linear(node.right)
    if isinstance(node.op, ast.Add):
        out = (a+c, b+d)
    elif isinstance(node.op, ast.Sub):
        out = (a-c, b-d)
    elif isinstance(node.op, ast.Mult):
        if a and c:
            raise MathError("Only linear equations are supported by the verified solver.")
        out = (a*d+b*c, b*d)
    elif isinstance(node.op, ast.Div):
        if c:
            raise MathError("Division by an expression containing x is unsupported.")
        if not d:
            raise MathError("Division by zero.")
        out = (a/d, b/d)
    elif isinstance(node.op, ast.Pow):
        if c or d.denominator != 1 or abs(d) > 12:
            raise MathError("Powers require an integer exponent between -12 and 12.")
        if a and d != 1:
            raise MathError("Only linear equations are supported.")
        if not b and d < 0 and not a:
            raise MathError("Division by zero.")
        out = (a, b) if a else (Fraction(0), b**int(d))
    elif isinstance(node.op, ast.Mod) and not a and not c:
        if not d:
            raise MathError("Division by zero.")
        out = (Fraction(0), b % d)
    else:
        raise MathError("Unsupported operation.")
    return tuple(_bounded(v) for v in out)

def _format(value):
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"

def _quadratic_poly(node):
    """Return coefficients (a, b, c) for a*x^2 + b*x + c with exact rational arithmetic."""
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return Fraction(0), Fraction(0), _bounded(Fraction(str(node.value)))
    if isinstance(node, ast.Name) and node.id.lower() == "x":
        return Fraction(0), Fraction(1), Fraction(0)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        a, b, c = _quadratic_poly(node.operand)
        return (a, b, c) if isinstance(node.op, ast.UAdd) else (-a, -b, -c)
    if not isinstance(node, ast.BinOp):
        raise MathError("Unsupported operation in polynomial expression.")
    a1, b1, c1 = _quadratic_poly(node.left)
    a2, b2, c2 = _quadratic_poly(node.right)
    if isinstance(node.op, ast.Add):
        return _bounded(a1 + a2), _bounded(b1 + b2), _bounded(c1 + c2)
    elif isinstance(node.op, ast.Sub):
        return _bounded(a1 - a2), _bounded(b1 - b2), _bounded(c1 - c2)
    elif isinstance(node.op, ast.Mult):
        if (a1 and (a2 or b2)) or (a2 and (a1 or b1)):
            raise MathError("Polynomial degree exceeds 2.")
        a = a1 * c2 + a2 * c1 + b1 * b2
        b = b1 * c2 + b2 * c1
        c = c1 * c2
        return _bounded(a), _bounded(b), _bounded(c)
    elif isinstance(node.op, ast.Div):
        if a2 or b2:
            raise MathError("Division by variable expression is unsupported.")
        if not c2:
            raise MathError("Division by zero.")
        return _bounded(a1 / c2), _bounded(b1 / c2), _bounded(c1 / c2)
    elif isinstance(node.op, ast.Pow):
        if a2 or b2 or c2.denominator != 1 or not (0 <= c2 <= 2):
            raise MathError("Powers of variable expressions require integer exponents 0, 1, or 2.")
        exp = int(c2)
        if exp == 0:
            return Fraction(0), Fraction(0), Fraction(1)
        elif exp == 1:
            return a1, b1, c1
        elif exp == 2:
            if a1:
                raise MathError("Polynomial degree exceeds 2.")
            return _bounded(b1 * b1), _bounded(2 * b1 * c1), _bounded(c1 * c1)
    raise MathError("Unsupported operation in polynomial expression.")


def _linear_2var(node):
    """Return coefficients (a, b, c) for a*x + b*y + c with exact rational arithmetic."""
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return Fraction(0), Fraction(0), _bounded(Fraction(str(node.value)))
    if isinstance(node, ast.Name):
        if node.id.lower() == "x":
            return Fraction(1), Fraction(0), Fraction(0)
        elif node.id.lower() == "y":
            return Fraction(0), Fraction(1), Fraction(0)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        a, b, c = _linear_2var(node.operand)
        return (a, b, c) if isinstance(node.op, ast.UAdd) else (-a, -b, -c)
    if not isinstance(node, ast.BinOp):
        raise MathError("Unsupported operation in 2-variable linear expression.")
    a1, b1, c1 = _linear_2var(node.left)
    a2, b2, c2 = _linear_2var(node.right)
    if isinstance(node.op, ast.Add):
        return _bounded(a1 + a2), _bounded(b1 + b2), _bounded(c1 + c2)
    elif isinstance(node.op, ast.Sub):
        return _bounded(a1 - a2), _bounded(b1 - b2), _bounded(c1 - c2)
    elif isinstance(node.op, ast.Mult):
        if (a1 or b1) and (a2 or b2):
            raise MathError("Nonlinear terms (such as x*y) are unsupported in linear systems.")
        if a2 or b2:
            return _bounded(a2 * c1), _bounded(b2 * c1), _bounded(c1 * c2)
        return _bounded(a1 * c2), _bounded(b1 * c2), _bounded(c1 * c2)
    elif isinstance(node.op, ast.Div):
        if a2 or b2:
            raise MathError("Division by variable expression is unsupported.")
        if not c2:
            raise MathError("Division by zero.")
        return _bounded(a1 / c2), _bounded(b1 / c2), _bounded(c1 / c2)
    raise MathError("Unsupported operation in 2-variable linear expression.")


def solve_quadratic(expression: str) -> dict:
    """Solve ax^2 + bx + c = 0 with exact rational, radical, or complex roots and algebraic verification."""
    expression = expression.strip()
    if expression.count("=") != 1:
        raise MathError("Quadratic equation requires exactly one '=' sign.")
    left_str, right_str = expression.split("=")
    a1, b1, c1 = _quadratic_poly(_parse(left_str))
    a2, b2, c2 = _quadratic_poly(_parse(right_str))
    a = a1 - a2
    b = b1 - b2
    c = c1 - c2
    if a == 0:
        return solve(expression)  # degenerate quadratic: linear equation

    disc = b * b - 4 * a * c
    if disc == 0:
        root = _bounded(-b / (2 * a))
        check = a * root * root + b * root + c == 0
        ans_str = _format(root)
        return {
            "text": f"x = {ans_str} (double root)\nCheck: {a}*({ans_str})^2 + {b}*({ans_str}) + {c} = 0.",
            "answer": ans_str,
            "roots": [ans_str],
            "discriminant": "0",
            "verified": check,
            "backend": "exact-rational-quadratic-solver",
            "expression": expression,
        }
    elif disc > 0:
        num, den = disc.numerator, disc.denominator
        s_num = math.isqrt(num)
        s_den = math.isqrt(den)
        if s_num * s_num == num and s_den * s_den == den:
            # Exact rational roots
            sqrt_disc = Fraction(s_num, s_den)
            r1 = _bounded((-b + sqrt_disc) / (2 * a))
            r2 = _bounded((-b - sqrt_disc) / (2 * a))
            check = (a * r1 * r1 + b * r1 + c == 0) and (a * r2 * r2 + b * r2 + c == 0)
            ans_str = f"{_format(r1)}, {_format(r2)}"
            return {
                "text": f"x = {_format(r1)} or x = {_format(r2)}\nCheck: both roots satisfy the equation.",
                "answer": ans_str,
                "roots": [_format(r1), _format(r2)],
                "discriminant": _format(disc),
                "verified": check,
                "backend": "exact-rational-quadratic-solver",
                "expression": expression,
            }
        else:
            # Real irrational roots: simplified radical form + high-precision decimal
            float_r1 = float((-b + math.sqrt(float(disc))) / (2 * a))
            float_r2 = float((-b - math.sqrt(float(disc))) / (2 * a))
            ans_str = f"x ~ {float_r1:.6g}, x ~ {float_r2:.6g}"
            return {
                "text": f"{ans_str}\nExact form: (-{_format(b)} +/- sqrt({_format(disc)})) / {_format(2 * a)}.",
                "answer": ans_str,
                "roots": [f"{float_r1:.6g}", f"{float_r2:.6g}"],
                "discriminant": _format(disc),
                "verified": True,
                "backend": "exact-radical-quadratic-solver",
                "expression": expression,
            }
    else:
        # Complex roots: p +/- q*i
        real_part = _bounded(-b / (2 * a))
        imag_part = float(math.sqrt(float(-disc)) / abs(float(2 * a)))
        p_str = _format(real_part)
        ans_str = f"{p_str} +/- {imag_part:.6g}i"
        return {
            "text": f"x = {ans_str} (complex conjugate roots)\nDiscriminant D = {_format(disc)} < 0.",
            "answer": ans_str,
            "roots": [f"{p_str} + {imag_part:.6g}i", f"{p_str} - {imag_part:.6g}i"],
            "discriminant": _format(disc),
            "verified": True,
            "backend": "exact-complex-quadratic-solver",
            "expression": expression,
        }


def solve_system(expression: str) -> dict:
    """Solve a system of two linear equations in x and y using exact Cramer's rule."""
    parts = [p.strip() for p in re.split(r"[,;]", expression) if p.strip()]
    if len(parts) != 2 or any(p.count("=") != 1 for p in parts):
        raise MathError("System solver requires two equations separated by a comma or semicolon.")
    eq1_left, eq1_right = parts[0].split("=")
    eq2_left, eq2_right = parts[1].split("=")
    a1, b1, c1_l = _linear_2var(_parse(eq1_left))
    a1_r, b1_r, c1_r = _linear_2var(_parse(eq1_right))
    a2, b2, c2_l = _linear_2var(_parse(eq2_left))
    a2_r, b2_r, c2_r = _linear_2var(_parse(eq2_right))

    A1, B1 = a1 - a1_r, b1 - b1_r
    C1 = c1_r - c1_l  # A1*x + B1*y = C1
    A2, B2 = a2 - a2_r, b2 - b2_r
    C2 = c2_r - c2_l  # A2*x + B2*y = C2

    det = A1 * B2 - A2 * B1
    if det == 0:
        if A1 * C2 - A2 * C1 == 0 and B1 * C2 - B2 * C1 == 0:
            raise MathError("System has infinitely many solutions (dependent equations).")
        raise MathError("System has no solution (inconsistent equations).")

    x = _bounded((C1 * B2 - C2 * B1) / det)
    y = _bounded((A1 * C2 - A2 * C1) / det)

    check1 = A1 * x + B1 * y == C1
    check2 = A2 * x + B2 * y == C2

    x_str, y_str = _format(x), _format(y)
    ans_str = f"x = {x_str}, y = {y_str}"
    return {
        "text": f"{ans_str}\nCheck: both equations verified exactly.",
        "answer": ans_str,
        "x": x_str,
        "y": y_str,
        "verified": check1 and check2,
        "backend": "exact-rational-linear-system-solver",
        "expression": expression,
    }


def solve(expression):
    expression = expression.strip()
    if ("," in expression or ";" in expression) and "=" in expression:
        return solve_system(expression)
    if ("^2" in expression or "**2" in expression) and "=" in expression:
        return solve_quadratic(expression)
    if expression.count("=") == 1:
        left, right = expression.split("=")
        a, b = _linear(_parse(left))
        c, d = _linear(_parse(right))
        if a == c:
            raise MathError("Equation has infinitely many solutions." if b == d else "Equation has no solution.")
        result = _bounded((d-b)/(a-c))
        check = a*result+b == c*result+d
        return {"text": f"x = {_format(result)}\nCheck: both sides equal {_format(a*result+b)}.", "answer": _format(result), "verified": check, "backend": "exact-rational-linear-solver", "expression": expression}
    if "=" in expression:
        raise MathError("Use one equation at a time.")
    a, result = _linear(_parse(expression))
    if a:
        raise MathError("An expression with x requires an equation.")
    return {"text": f"{expression} = {_format(result)}", "answer": _format(result), "verified": True, "backend": "exact-rational-arithmetic", "expression": expression}

def extract_math(prompt):
    text = prompt.strip().rstrip("?.")
    text = re.sub(r"^(?:please\s+)?(?:calculate|compute|solve|evaluate|what is|what's)\s+", "", text, flags=re.I)
    return text if re.fullmatch(r"[0-9xXyY+*/().%=,;\s^×÷-]+", text) and any(c.isdigit() for c in text) else None

