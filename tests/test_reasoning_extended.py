import pytest
from poseidon.reasoning import solve, MathError, extract_math, solve_linear_system, solve_polynomial


def test_polynomial_cubic_three_rational_roots():
    res = solve("x^3 - 6*x^2 + 11*x - 6 = 0")
    assert res["roots"] == ["1", "2", "3"]
    assert res["verified"] is True
    assert res["exact"] is True
    assert res["backend"] == "exact-rational-polynomial-solver"


def test_polynomial_factored_input():
    res = solve("(x - 1)*(x + 2)*(x - 3) = 0")
    assert "1" in res["roots"]
    assert "-2" in res["roots"]
    assert "3" in res["roots"]
    assert res["verified"] is True


def test_polynomial_repeated_root():
    res = solve("x^3 - 3*x^2 + 3*x - 1 = 0")
    assert res["roots"] == ["1"]
    assert res["multiplicities"] == [3]
    assert res["verified"] is True


def test_polynomial_quartic_mixed():
    res = solve("x^4 - 16 = 0")
    assert "2" in res["roots"]
    assert "-2" in res["roots"]
    assert res["verified"] is True


def test_polynomial_degree_rejection():
    with pytest.raises(MathError, match=r"(exponents between 0 and 8|degree exceeds)"):
        solve("x^9 = 1")
    with pytest.raises(MathError, match=r"(exponents between 0 and 8|degree exceeds)"):
        solve("x^5 * x^4 = 1")


def test_linear_system_3x3_gaussian():
    res = solve("x + y + z = 6, 2*x - y + z = 3, x + 2*y - z = 2")
    assert res["x"] == "1"
    assert res["y"] == "2"
    assert res["z"] == "3"
    assert res["verified"] is True
    assert res["backend"] == "exact-rational-gaussian-elimination"


def test_linear_system_4x4_gaussian():
    eqs = "x + y + z + w = 10, x - y + z - w = 0, 2*x + y - z = 3, x + 2*w = 7"
    res = solve(eqs)
    assert res["verified"] is True
    assert res["backend"] == "exact-rational-gaussian-elimination"
    assert set(res["variables"]) == {"x", "y", "z", "w"}


def test_linear_system_inconsistent_3x3():
    with pytest.raises(MathError, match="no solution"):
        solve("x + y + z = 1, x + y + z = 2, 2*x + y = 3")


def test_linear_system_dependent_3x3():
    with pytest.raises(MathError, match="infinitely many"):
        solve("x + y + z = 3, 2*x + 2*y + 2*z = 6, x - y = 0")


def test_extract_math_extended():
    assert extract_math("solve x^3 - 8 = 0") == "x^3 - 8 = 0"
    assert extract_math("compute x + y + z = 6, x - y = 0, y - z = -1") == "x + y + z = 6, x - y = 0, y - z = -1"
    assert extract_math("What are you doing today?") is None
    assert extract_math("How are you?") is None
