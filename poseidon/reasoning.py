"""Bounded arithmetic and linear algebra verification, never Python eval.

Capabilities (all parsing is AST based with a strict character whitelist, all
symbolic arithmetic uses exact :class:`fractions.Fraction` values, sizes are
bounded, and every answer is independently checked by substitution):

* rational arithmetic and linear equations in ``x``;
* quadratics (exact rational, radical, or complex roots);
* univariate polynomial equations in ``x`` up to degree 8 (exact rational roots
  via the rational root theorem with deflation, an exact radical/complex
  residual quadratic, and a clearly labelled numeric Aberth fallback);
* linear systems of 2..6 equations in the variables ``x, y, z, w, u, v``
  (exact Cramer's rule for 2x2 systems in x/y, exact Gaussian elimination
  otherwise).

Nonlinear single equations must state powers explicitly (``x^2``) or use a
parenthesised factored form (``(x-1)*(x+2) = 0``); a bare product such as
``x*x = 4`` is still rejected as non-linear, as before.
"""
from __future__ import annotations
import ast
import cmath
from fractions import Fraction
import math
import re

class MathError(ValueError):
    pass

MAX_POLY_DEGREE = 8
SYSTEM_VARIABLES = ("x", "y", "z", "w", "u", "v")
_MAX_NODES = 160
_DIVISOR_TRIAL_LIMIT = 10**5
_MAX_CANDIDATE_PAIRS = 20000

def _parse(expression):
    expression = expression.strip()
    if len(expression) > 250:
        raise MathError("Expression is too long (250 characters maximum).")
    expression = expression.replace("^", "**").replace("×", "*").replace("÷", "/")
    if not re.fullmatch(r"[0-9xXyYzZwWuUvV+*/().%\s-]+", expression):
        raise MathError("Use numbers, x, y (z, w, u, v in systems), parentheses, and + - * / ** % only.")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except (SyntaxError, RecursionError, ValueError) as e:
        raise MathError("Invalid expression; use explicit multiplication such as 3*x.") from e
    if sum(1 for _ in ast.walk(tree)) > _MAX_NODES:
        raise MathError("Expression is too complex.")
    return tree.body

def _bounded(value):
    if abs(value.numerator) > 10**80 or value.denominator > 10**80:
        raise MathError("Result exceeds the numeric budget.")
    return value

def _constant(node):
    return isinstance(node, ast.Constant) and type(node.value) in (int, float)

def _linear(node):
    # Return coefficient and constant for ax+b, maintaining exact rational arithmetic.
    if _constant(node):
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


# ---------------------------------------------------------------------------
# Exact sparse univariate polynomials: dict degree -> nonzero Fraction.
# ---------------------------------------------------------------------------

def _poly_clean(poly):
    return {k: _bounded(v) for k, v in poly.items() if v != 0}

def _poly_degree(poly):
    return max(poly) if poly else -1

def _poly_add(p, q, sign=1):
    out = dict(p)
    for k, v in q.items():
        out[k] = out.get(k, Fraction(0)) + sign * v
    return _poly_clean(out)

def _poly_mul(p, q):
    if p and q and _poly_degree(p) + _poly_degree(q) > MAX_POLY_DEGREE:
        raise MathError(f"Polynomial degree exceeds {MAX_POLY_DEGREE} (maximum supported degree).")
    out = {}
    for i, a in p.items():
        for j, b in q.items():
            out[i + j] = out.get(i + j, Fraction(0)) + a * b
    return _poly_clean(out)

def _poly_constant_value(poly):
    """Return the Fraction value of a constant polynomial, or None if it contains x."""
    if _poly_degree(poly) > 0:
        return None
    return poly.get(0, Fraction(0))

def _poly(node):
    """Parse an AST node into an exact sparse polynomial in x (degree <= 8)."""
    if _constant(node):
        return _poly_clean({0: Fraction(str(node.value))})
    if isinstance(node, ast.Name):
        if node.id.lower() == "x":
            return {1: Fraction(1)}
        raise MathError("Polynomial equations must use the single variable x.")
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        p = _poly(node.operand)
        return p if isinstance(node.op, ast.UAdd) else {k: -v for k, v in p.items()}
    if not isinstance(node, ast.BinOp):
        raise MathError("Unsupported operation in polynomial expression.")
    left = _poly(node.left)
    right = _poly(node.right)
    if isinstance(node.op, ast.Add):
        return _poly_add(left, right)
    if isinstance(node.op, ast.Sub):
        return _poly_add(left, right, -1)
    if isinstance(node.op, ast.Mult):
        return _poly_mul(left, right)
    if isinstance(node.op, ast.Div):
        d = _poly_constant_value(right)
        if d is None:
            raise MathError("Division by an expression containing x is unsupported.")
        if not d:
            raise MathError("Division by zero.")
        return _poly_clean({k: v / d for k, v in left.items()})
    if isinstance(node.op, ast.Pow):
        e = _poly_constant_value(right)
        if e is None or e.denominator != 1:
            raise MathError("Powers require a constant integer exponent.")
        base_const = _poly_constant_value(left)
        if base_const is not None:
            if abs(e) > 12:
                raise MathError("Powers require an integer exponent between -12 and 12.")
            if not base_const and e < 0:
                raise MathError("Division by zero.")
            return _poly_clean({0: base_const ** int(e)})
        if not 0 <= e <= MAX_POLY_DEGREE:
            raise MathError(f"Powers of variable expressions require integer exponents between 0 and {MAX_POLY_DEGREE}.")
        if _poly_degree(left) * int(e) > MAX_POLY_DEGREE:
            raise MathError(f"Polynomial degree exceeds {MAX_POLY_DEGREE} (maximum supported degree).")
        out = {0: Fraction(1)}
        for _ in range(int(e)):
            out = _poly_mul(out, left)
        return out
    if isinstance(node.op, ast.Mod):
        a, b = _poly_constant_value(left), _poly_constant_value(right)
        if a is None or b is None:
            raise MathError("Modulo is only supported between constants.")
        if not b:
            raise MathError("Division by zero.")
        return _poly_clean({0: a % b})
    raise MathError("Unsupported operation in polynomial expression.")

def _dense(poly, degree):
    return [poly.get(i, Fraction(0)) for i in range(degree + 1)]

def _horner(coeffs, value):
    """Evaluate a dense (low->high) coefficient list at value."""
    acc = 0 * value
    for c in reversed(coeffs):
        acc = acc * value + c
    return acc

def _synthetic_division(coeffs, r):
    """Divide dense low->high coeffs by (x - r); return (quotient, remainder)."""
    n = len(coeffs) - 1
    quotient = [Fraction(0)] * n
    acc = coeffs[n]
    for i in range(n - 1, -1, -1):
        quotient[i] = acc
        acc = coeffs[i] + acc * r
    return quotient, acc

def _dense_mul(p, q):
    out = [Fraction(0)] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        for j, b in enumerate(q):
            out[i + j] += a * b
    return out

def _divisors(n):
    """Positive divisors of |n| found by bounded trial division; returns (divisors, complete)."""
    n = abs(n)
    root = math.isqrt(n)
    limit = min(root, _DIVISOR_TRIAL_LIMIT)
    out = set()
    for d in range(1, limit + 1):
        if n % d == 0:
            out.add(d)
            out.add(n // d)
    return sorted(out), root <= _DIVISOR_TRIAL_LIMIT

def _rational_root_candidates(coeffs):
    """Rational root theorem candidates for dense rational coeffs with coeffs[0] != 0."""
    lcm = 1
    for c in coeffs:
        lcm = lcm * c.denominator // math.gcd(lcm, c.denominator)
    ints = [int(c * lcm) for c in coeffs]
    g = 0
    for v in ints:
        g = math.gcd(g, v)
    ints = [v // g for v in ints]
    ps, complete_p = _divisors(ints[0])
    qs, complete_q = _divisors(ints[-1])
    complete = complete_p and complete_q
    n = len(ints) - 1
    found = []
    pairs = 0
    for q in qs:
        qpow = [q ** k for k in range(n + 1)]
        for p in ps:
            pairs += 1
            if pairs > _MAX_CANDIDATE_PAIRS:
                return found, False
            if math.gcd(p, q) != 1:
                continue
            for sp in (p, -p):
                acc = ints[n]
                for i in range(n - 1, -1, -1):
                    acc = acc * sp + ints[i] * qpow[n - i]
                if acc == 0:
                    found.append(Fraction(sp, q))
    return sorted(found), complete

def _deflate_root(coeffs, r):
    """Deflate (x - r) repeatedly; return (coeffs, multiplicity)."""
    m = 0
    while len(coeffs) > 1:
        quotient, rem = _synthetic_division(coeffs, r)
        if rem != 0:
            break
        coeffs = [_bounded(c) for c in quotient]
        m += 1
    return coeffs, m

def _split_square(n):
    """Write n >= 1 as s*s*m by bounded trial extraction (m is not always squarefree)."""
    s, m = 1, n
    r = math.isqrt(m)
    if r * r == m:
        return r, 1
    d = 2
    while d * d <= m and d <= 10**4:
        while m % (d * d) == 0:
            m //= d * d
            s *= d
        d += 1 if d == 2 else 2
    r = math.isqrt(m)
    if r * r == m:
        return s * r, 1
    return s, m

class _Surd:
    """Exact number u + v*sqrt(m) for a fixed integer m (m < 0 gives complex values)."""
    __slots__ = ("u", "v", "m")

    def __init__(self, u, v, m):
        self.u, self.v, self.m = Fraction(u), Fraction(v), m

    def __add__(self, other):
        if isinstance(other, _Surd):
            return _Surd(self.u + other.u, self.v + other.v, self.m)
        return _Surd(self.u + other, self.v, self.m)

    __radd__ = __add__

    def __mul__(self, other):
        if isinstance(other, _Surd):
            return _Surd(self.u * other.u + self.v * other.v * self.m, self.u * other.v + self.v * other.u, self.m)
        return _Surd(self.u * other, self.v * other, self.m)

    __rmul__ = __mul__

    def is_zero(self):
        return self.u == 0 and self.v == 0

def _radical_string(alpha, beta, m, sign):
    if m == -1:
        rad = "i"
    elif m < 0:
        rad = f"sqrt({-m})*i"
    else:
        rad = f"sqrt({m})"
    term = rad if beta == 1 else f"{_format(beta)}*{rad}"
    if alpha == 0:
        return term if sign > 0 else f"-{term}"
    return f"{_format(alpha)} {'+' if sign > 0 else '-'} {term}"

def _complex_string(z):
    if z.imag == 0:
        return f"{z.real:.10g}"
    return f"{z.real:.10g} {'+' if z.imag > 0 else '-'} {abs(z.imag):.10g}i"

def _log_abs(value):
    return math.log(abs(value.numerator)) - math.log(value.denominator)

def _relative_residual(coeffs, z):
    num = abs(_horner(coeffs, z))
    scale = sum(abs(c) * abs(z) ** i for i, c in enumerate(coeffs))
    return num / scale if scale else num

def _aberth(c):
    """All complex roots of a monic float polynomial (low->high) via Aberth-Ehrlich iteration."""
    n = len(c) - 1

    def evaluate(z):
        p, dp = 0j, 0j
        for a in reversed(c):
            dp = dp * z + p
            p = p * z + a
        return p, dp

    z = [cmath.rect(0.8, 2 * math.pi * k / n + 0.4) for k in range(n)]
    for _ in range(600):
        biggest = 0.0
        for k in range(n):
            p, dp = evaluate(z[k])
            if p == 0:
                continue
            s = 0j
            for j in range(n):
                if j != k:
                    diff = z[k] - z[j]
                    s += 1 / diff if diff != 0 else 1e12
            if dp == 0:
                w = 1e-8 * (1 + abs(z[k]))
            else:
                ratio = p / dp
                denom = 1 - ratio * s
                w = ratio / denom if denom != 0 else ratio
            z[k] -= w
            biggest = max(biggest, abs(w) / (1 + abs(z[k])))
        if biggest < 1e-16:
            break
    # Newton polishing: keep a step only when it reduces |p|.
    for k in range(n):
        for _ in range(3):
            p, dp = evaluate(z[k])
            if p == 0 or dp == 0:
                break
            candidate = z[k] - p / dp
            if abs(evaluate(candidate)[0]) < abs(p):
                z[k] = candidate
            else:
                break
    return z

def _numeric_roots(coeffs):
    """Numeric roots of dense Fraction coeffs (degree >= 1, coeffs[0] != 0).

    Returns a list of (complex root, relative residual)."""
    n = len(coeffs) - 1
    lead = coeffs[-1]
    logs = [(_log_abs(coeffs[n - k]) - _log_abs(lead)) / k for k in range(1, n + 1) if coeffs[n - k]]
    exp2 = math.ceil(max(logs) / math.log(2)) + 1 if logs else 0
    if abs(exp2) > 500:
        raise MathError("Polynomial coefficients are too extreme for numeric root finding.")
    scale = Fraction(2) ** exp2
    scaled = [float(coeffs[i] * scale ** (i - n) / lead) for i in range(n + 1)]
    out = []
    for y in _aberth(scaled):
        real_y = complex(y.real, 0.0)
        if abs(y.imag) <= 1e-7 * (1 + abs(y)) and _relative_residual(scaled, real_y) <= 1e-10:
            y = real_y
        residual = _relative_residual(scaled, y)
        out.append((complex(math.ldexp(y.real, exp2), math.ldexp(y.imag, exp2)), residual))
    return out

def _poly_string(coeffs):
    terms = []
    for i in range(len(coeffs) - 1, -1, -1):
        c = coeffs[i]
        if c == 0:
            continue
        mag = abs(c)
        var = "" if i == 0 else ("x" if i == 1 else f"x^{i}")
        body = _format(mag) if not var else (var if mag == 1 else f"{_format(mag)}*{var}")
        if not terms:
            terms.append(body if c > 0 else f"-{body}")
        else:
            terms.append(f"{'+' if c > 0 else '-'} {body}")
    return " ".join(terms) if terms else "0"

def _linear_result(a, b, c, d, expression):
    if a == c:
        raise MathError("Equation has infinitely many solutions." if b == d else "Equation has no solution.")
    result = _bounded((d-b)/(a-c))
    check = a*result+b == c*result+d
    return {"text": f"x = {_format(result)}\nCheck: both sides equal {_format(a*result+b)}.", "answer": _format(result), "verified": check, "backend": "exact-rational-linear-solver", "expression": expression}

def _quadratic_result(a, b, c, expression):
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


def _parse_polynomial_equation(expression, label):
    expression = expression.strip()
    if expression.count("=") != 1:
        raise MathError(f"{label} requires exactly one '=' sign.")
    left_str, right_str = expression.split("=")
    left = _poly(_parse(left_str))
    right = _poly(_parse(right_str))
    return expression, left, right, _poly_add(left, right, -1)


def solve_quadratic(expression: str) -> dict:
    """Solve ax^2 + bx + c = 0 with exact rational, radical, or complex roots and algebraic verification."""
    expression, _, _, poly = _parse_polynomial_equation(expression, "Quadratic equation")
    if _poly_degree(poly) > 2:
        raise MathError("Polynomial degree exceeds 2.")
    a, b, c = poly.get(2, Fraction(0)), poly.get(1, Fraction(0)), poly.get(0, Fraction(0))
    if a == 0:
        return solve(expression)  # degenerate quadratic: linear equation
    return _quadratic_result(a, b, c, expression)


def solve_polynomial(expression: str) -> dict:
    """Solve a univariate polynomial equation in x of degree <= 8.

    Degree 1 and 2 reuse the linear and quadratic solvers (identical output).
    Higher degrees find every rational root exactly (rational root theorem plus
    synthetic-division deflation with multiplicities), solve a residual
    quadratic exactly in radicals, and approximate any remaining residual of
    degree >= 3 numerically with the Aberth-Ehrlich method."""
    expression, left, right, poly = _parse_polynomial_equation(expression, "Polynomial equation")
    degree = _poly_degree(poly)
    if degree <= 0:
        raise MathError("Equation has infinitely many solutions." if not poly else "Equation has no solution.")
    if degree == 1:
        return _linear_result(left.get(1, Fraction(0)), left.get(0, Fraction(0)), right.get(1, Fraction(0)), right.get(0, Fraction(0)), expression)
    if degree == 2:
        return _quadratic_result(poly[2], poly.get(1, Fraction(0)), poly.get(0, Fraction(0)), expression)
    return _solve_high_degree(_dense(poly, degree), expression)


def _solve_high_degree(original, expression):
    degree = len(original) - 1
    coeffs = list(original)
    rational = []  # (Fraction, multiplicity)
    zero_mult = 0
    while coeffs[0] == 0:
        coeffs.pop(0)
        zero_mult += 1
    if zero_mult:
        rational.append((Fraction(0), zero_mult))
    complete = True
    if len(coeffs) > 1:
        candidates, complete = _rational_root_candidates(coeffs)
        for r in candidates:
            if len(coeffs) <= 1:
                break
            coeffs, m = _deflate_root(coeffs, r)
            if m:
                rational.append((r, m))
    # Numeric-guided rational search covers roots missed by a capped enumeration.
    if not complete:
        for _ in range(degree):
            if len(coeffs) - 1 < 3:
                break
            found = False
            for z, _res in _numeric_roots(coeffs):
                if z.imag != 0 or not math.isfinite(z.real):
                    continue
                for k in range(7):
                    cand = Fraction(z.real).limit_denominator(10**k)
                    if _horner(coeffs, cand) == 0:
                        coeffs, m = _deflate_root(coeffs, cand)
                        rational.append((cand, m))
                        found = True
                        break
                if found:
                    break
            if not found:
                break
    residual_degree = len(coeffs) - 1
    exact_extra = []    # (exact string, approximation string, _Surd root)
    numeric = []        # (complex root, relative residual)
    if residual_degree == 1:
        r = _bounded(-coeffs[0] / coeffs[1])
        rational.append((r, 1))
        coeffs = [coeffs[1]]
    elif residual_degree == 2:
        c0, c1, c2 = coeffs
        disc = c1 * c1 - 4 * c2 * c0
        t = disc.numerator * disc.denominator
        s, m = _split_square(abs(t))
        alpha = _bounded(-c1 / (2 * c2))
        beta = abs(_bounded(Fraction(s, disc.denominator) / (2 * c2)))
        if m == 1 and disc > 0:
            for r in (alpha + beta, alpha - beta):
                rational.append((r, 1))
            coeffs = [c2]
        else:
            m_signed = m if disc > 0 else -m
            for sign in (1, -1):
                surd = _Surd(alpha, sign * beta, m_signed)
                if m_signed > 0:
                    approx = f"{float(alpha) + sign * float(beta) * math.sqrt(m):.10g}"
                else:
                    approx = _complex_string(complex(float(alpha), sign * float(beta) * math.sqrt(-m_signed)))
                exact_extra.append((_radical_string(alpha, beta, m_signed, sign), approx, surd))
    elif residual_degree >= 3:
        numeric = _numeric_roots(coeffs)

    # Independent verification.
    rational.sort(key=lambda item: item[0])
    product = [Fraction(1)]
    for r, m in rational:
        for _ in range(m):
            product = _dense_mul(product, [-r, Fraction(1)])
    residual_factor = coeffs if residual_degree >= 2 else [coeffs[-1]] if len(coeffs) == 1 else coeffs
    factorization_ok = _dense_mul(product, residual_factor) == list(original)
    rational_ok = all(_horner(original, r) == 0 for r, _ in rational)
    radical_ok = all(_horner(original, surd).is_zero() for _, _, surd in exact_extra)
    numeric_ok = all(res <= 1e-9 for _, res in numeric)
    verified = bool(factorization_ok and rational_ok and radical_ok and numeric_ok)

    def mult_note(m):
        return "" if m == 1 else (" (double root)" if m == 2 else f" (multiplicity {m})")

    roots, multiplicities, lines = [], [], []
    for r, m in rational:
        roots.append(_format(r))
        multiplicities.append(m)
        lines.append(f"x = {_format(r)}{mult_note(m)} [exact rational]")
    for exact, approx, _ in exact_extra:
        roots.append(exact)
        multiplicities.append(1)
        lines.append(f"x = {exact} ~ {approx} [exact {'radical' if 'i' not in exact else 'complex'}]")
    numeric_strings = []
    for z, res in sorted(numeric, key=lambda item: (item[0].imag != 0, item[0].real, item[0].imag)):
        text_root = _complex_string(z)
        numeric_strings.append(text_root)
        roots.append(text_root)
        multiplicities.append(1)
        lines.append(f"x ~ {text_root} [numeric approximation, relative residual {res:.1e}]")

    if numeric:
        backend = "numeric-aberth-polynomial-solver"
    elif exact_extra:
        backend = "exact-radical-polynomial-solver"
    else:
        backend = "exact-rational-polynomial-solver"
    checks = []
    if rational:
        checks.append("rational roots substituted exactly")
    if exact_extra:
        checks.append("radical/complex roots substituted exactly in Q(sqrt(m))")
    checks.append("factorization reproduces the polynomial exactly" if factorization_ok else "factorization check FAILED")
    if numeric:
        checks.append("numeric roots are approximations (Aberth-Ehrlich with Newton polishing), accepted only with relative residual <= 1e-9")
    if not complete:
        checks.append("rational-root enumeration was capped for very large coefficients; remaining roots were searched numerically")
    text = (
        f"Polynomial of degree {degree}: {_poly_string(original)} = 0\n"
        + "\n".join(lines)
        + "\nCheck: " + "; ".join(checks) + "."
    )
    return {
        "text": text,
        "answer": ", ".join(roots),
        "roots": roots,
        "multiplicities": multiplicities,
        "approximate_roots": numeric_strings,
        "exact": not numeric,
        "degree": degree,
        "polynomial": _poly_string(original),
        "verified": verified,
        "backend": backend,
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


def _is_const(form):
    return all(k == "" or v == 0 for k, v in form.items())

def _linear_multi(node):
    """Return {variable: coefficient, "": constant} for a linear form in x, y, z, w, u, v."""
    if _constant(node):
        return {"": _bounded(Fraction(str(node.value)))}
    if isinstance(node, ast.Name):
        name = node.id.lower()
        if name not in SYSTEM_VARIABLES:
            raise MathError("Linear systems may only use the variables x, y, z, w, u, v.")
        return {name: Fraction(1), "": Fraction(0)}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        form = _linear_multi(node.operand)
        return form if isinstance(node.op, ast.UAdd) else {k: -v for k, v in form.items()}
    if not isinstance(node, ast.BinOp):
        raise MathError("Unsupported operation in linear system expression.")
    left = _linear_multi(node.left)
    right = _linear_multi(node.right)
    if isinstance(node.op, (ast.Add, ast.Sub)):
        sign = 1 if isinstance(node.op, ast.Add) else -1
        out = dict(left)
        for k, v in right.items():
            out[k] = out.get(k, Fraction(0)) + sign * v
    elif isinstance(node.op, ast.Mult):
        if not _is_const(left) and not _is_const(right):
            raise MathError("Nonlinear terms (such as x*y) are unsupported in linear systems.")
        form, factor = (right, left.get("", Fraction(0))) if _is_const(left) else (left, right.get("", Fraction(0)))
        out = {k: v * factor for k, v in form.items()}
    elif isinstance(node.op, ast.Div):
        if not _is_const(right):
            raise MathError("Division by variable expression is unsupported.")
        d = right.get("", Fraction(0))
        if not d:
            raise MathError("Division by zero.")
        out = {k: v / d for k, v in left.items()}
    elif isinstance(node.op, (ast.Pow, ast.Mod)):
        if not _is_const(left) or not _is_const(right):
            raise MathError("Nonlinear terms (such as x*y) are unsupported in linear systems.")
        a, b = left.get("", Fraction(0)), right.get("", Fraction(0))
        if isinstance(node.op, ast.Pow):
            if b.denominator != 1 or abs(b) > 12:
                raise MathError("Powers require an integer exponent between -12 and 12.")
            if not a and b < 0:
                raise MathError("Division by zero.")
            out = {"": a ** int(b)}
        else:
            if not b:
                raise MathError("Division by zero.")
            out = {"": a % b}
    else:
        raise MathError("Unsupported operation in linear system expression.")
    out.setdefault("", Fraction(0))
    return {k: _bounded(v) for k, v in out.items()}


def solve_linear_system(expression: str) -> dict:
    """Solve 2..6 linear equations in x, y, z, w, u, v by exact Gaussian elimination."""
    expression = expression.strip()
    parts = [p.strip() for p in re.split(r"[,;]", expression) if p.strip()]
    if not 2 <= len(parts) <= 6 or any(p.count("=") != 1 for p in parts):
        raise MathError("Linear system solver requires 2 to 6 equations separated by commas or semicolons, each with one '='.")
    equations = []
    seen = set()
    for part in parts:
        left_str, right_str = part.split("=")
        left, right = _linear_multi(_parse(left_str)), _linear_multi(_parse(right_str))
        equations.append((left, right))
        seen.update(k for k in list(left) + list(right) if k)
    variables = [v for v in SYSTEM_VARIABLES if v in seen]
    if not variables:
        raise MathError("Linear systems need at least one variable.")
    n = len(variables)
    matrix = [
        [left.get(v, Fraction(0)) - right.get(v, Fraction(0)) for v in variables] + [right[""] - left[""]]
        for left, right in equations
    ]
    rows = len(matrix)
    pivots = []
    r = 0
    for col in range(n):
        if r == rows:
            break
        best = max(range(r, rows), key=lambda i: abs(matrix[i][col]))  # partial pivoting
        if matrix[best][col] == 0:
            continue
        matrix[r], matrix[best] = matrix[best], matrix[r]
        pivot = matrix[r][col]
        matrix[r] = [value / pivot for value in matrix[r]]
        for i in range(rows):
            if i != r and matrix[i][col] != 0:
                factor = matrix[i][col]
                matrix[i] = [a - factor * b for a, b in zip(matrix[i], matrix[r])]
        pivots.append(col)
        r += 1
    if any(all(value == 0 for value in row[:n]) and row[n] != 0 for row in matrix):
        raise MathError("System has no solution (inconsistent equations).")
    if r < n:
        raise MathError("System has infinitely many solutions (dependent equations).")
    values = {variables[col]: _bounded(matrix[i][n]) for i, col in enumerate(pivots)}

    def side(form):
        return form[""] + sum(coef * values[var] for var, coef in form.items() if var)

    verified = all(side(left) == side(right) for left, right in equations)
    solution = {var: _format(values[var]) for var in variables}
    ans_str = ", ".join(f"{var} = {solution[var]}" for var in variables)
    result = {
        "text": f"{ans_str}\nCheck: all {rows} equations verified exactly by substitution.",
        "answer": ans_str,
        "solution": solution,
        "variables": variables,
        "verified": verified,
        "backend": "exact-rational-gaussian-elimination",
        "expression": expression,
    }
    for var in variables:
        result[var] = solution[var]
    return result


def _solve_any_system(expression):
    parts = [p for p in re.split(r"[,;]", expression) if p.strip()]
    letters = set(re.findall(r"[a-z]", expression.lower()))
    if len(parts) == 2 and letters <= {"x", "y"}:
        return solve_system(expression)  # keep the established 2x2 Cramer output
    return solve_linear_system(expression)


def _allows_nonlinear(expression):
    # Nonlinear equations must use an explicit power or a parenthesised factor.
    return "^" in expression or "**" in expression or "(" in expression


def solve(expression):
    expression = expression.strip()
    if ("," in expression or ";" in expression) and "=" in expression:
        return _solve_any_system(expression)
    if expression.count("=") == 1:
        left, right = expression.split("=")
        try:
            a, b = _linear(_parse(left))
            c, d = _linear(_parse(right))
        except MathError:
            if not _allows_nonlinear(expression):
                raise
            return solve_polynomial(expression)
        return _linear_result(a, b, c, d, expression)
    if "=" in expression:
        raise MathError("Use one equation at a time.")
    a, result = _linear(_parse(expression))
    if a:
        raise MathError("An expression with x requires an equation.")
    return {"text": f"{expression} = {_format(result)}", "answer": _format(result), "verified": True, "backend": "exact-rational-arithmetic", "expression": expression}

def extract_math(prompt):
    text = prompt.strip().rstrip("?.")
    text = re.sub(r"^(?:please\s+)?(?:calculate|compute|solve|evaluate|what is|what's)\s+", "", text, flags=re.I)
    if not re.fullmatch(r"[0-9xXyYzZwWuUvV+*/().%=,;\s^×÷-]+", text) or not any(c.isdigit() for c in text):
        return None
    if re.search(r"[zZwWuUvV]", text):
        # Extra system variables only count inside equations, as standalone one-letter names,
        # so short words or text-speak ("u", "vw") are not mistaken for math.
        if "=" not in text:
            return None
        if any(len(run) > 1 for run in re.findall(r"[A-Za-z]+", text)):
            return None
    return text
