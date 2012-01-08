from sympy import (Symbol, symbols, nan, oo, Float, conjugate, sqrt, sin, cos, pi, re, im, Abs, O, I, S,
                  factorial, erf, fresnels, fresnelc)

from sympy.core.function import ArgumentIndexError

from sympy.utilities.pytest import raises

x, y, z = symbols('x,y,z')
w = Symbol("w", real=True)
n = Symbol("n", integer=True)

def test_erf():
    assert erf(nan) == nan

    assert erf(oo) == 1
    assert erf(-oo) == -1

    assert erf(0) == 0

    assert erf(-2) == -erf(2)
    assert erf(-x*y) == -erf(x*y)
    assert erf(-x - y) == -erf(x + y)

    assert erf(I).is_real == False
    assert erf(0).is_real == True

    assert erf(x).as_leading_term(x) == x
    assert erf(1/x).as_leading_term(x) == erf(1/x)

    raises(ArgumentIndexError, 'erf(x).fdiff(2)')

def test_erf_series():
    assert erf(x).series(x, 0, 7) == 2*x/sqrt(pi) - \
        2*x**3/3/sqrt(pi) + x**5/5/sqrt(pi) + O(x**7)

def test_erf_evalf():
    assert abs( erf(Float(2.0)) - 0.995322265 )  <  1E-8  # XXX

def test_fresnel():
    assert fresnels(0) == 0
    assert fresnels(oo) == S.Half
    assert fresnels(-oo) == -S.Half

    assert fresnels(z) == fresnels(z)
    assert fresnels(-z) == -fresnels(z)
    assert fresnels(I*z) == -I*fresnels(z)
    assert fresnels(-I*z) == I*fresnels(z)

    assert conjugate(fresnels(z)) == fresnels(conjugate(z))

    assert fresnels(z).diff(z) == sin(pi*z**2/2)

    assert fresnels(z).as_leading_term(z) == pi*z**3/6

    assert fresnels(z).rewrite(erf) == (S.One+I)/4 * (erf((S.One+I)/2*sqrt(pi)*z) - I*erf((S.One-I)/2*sqrt(pi)*z))

    assert fresnels(z)._eval_nseries(z, n, None) == z**3*(-z**4)**n*(2**(-2*n-1)*pi**(2*n+1))/((4*n+3)*factorial(2*n+1))

    assert fresnels(z)._eval_aseries(z, oo, 0, 0) == S.Half - cos(pi*z**2/2)/(pi*z)

    assert fresnels(w).is_real is True

    assert fresnels(z).as_real_imag() == ((fresnels(re(z) - I*re(z)*Abs(im(z))/Abs(re(z)))/2 + fresnels(re(z) + I*re(z)*Abs(im(z))/Abs(re(z)))/2,
                                          I*(fresnels(re(z) - I*re(z)*Abs(im(z))/Abs(re(z))) - fresnels(re(z) + I*re(z)*Abs(im(z))/Abs(re(z))))*
                                          re(z)*Abs(im(z))/(2*im(z)*Abs(re(z)))))

    assert fresnelc(0) == 0
    assert fresnelc(oo) == S.Half
    assert fresnelc(-oo) == -S.Half

    assert fresnelc(z) == fresnelc(z)
    assert fresnelc(-z) == -fresnelc(z)
    assert fresnelc(I*z) == I*fresnelc(z)
    assert fresnelc(-I*z) == -I*fresnelc(z)

    assert conjugate(fresnelc(z)) == fresnelc(conjugate(z))

    assert fresnelc(z).diff(z) == cos(pi*z**2/2)

    assert fresnelc(z).as_leading_term(z) == z

    assert fresnelc(z).rewrite(erf) == (S.One-I)/4 * (erf((S.One+I)/2*sqrt(pi)*z) + I*erf((S.One-I)/2*sqrt(pi)*z))

    assert fresnelc(z)._eval_nseries(z, n, None) ==  z*(-z**4)**n*(2**(-2*n)*pi**(2*n))/((4*n+1)*factorial(2*n))

    assert fresnelc(z)._eval_aseries(z, oo, 0, 0) == S.Half + sin(pi*z**2/2)/(pi*z)

    assert fresnelc(w).is_real is True

    assert fresnelc(z).as_real_imag() == ((fresnelc(re(z) - I*re(z)*Abs(im(z))/Abs(re(z)))/2 + fresnelc(re(z) + I*re(z)*Abs(im(z))/Abs(re(z)))/2,
                                           I*(fresnelc(re(z) - I*re(z)*Abs(im(z))/Abs(re(z))) - fresnelc(re(z) + I*re(z)*Abs(im(z))/Abs(re(z))))*
                                           re(z)*Abs(im(z))/(2*im(z)*Abs(re(z)))))
