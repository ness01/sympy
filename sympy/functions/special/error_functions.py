from sympy.core import S, C, sympify, cacheit, pi, I
from sympy.core.function import Function, ArgumentIndexError
from sympy.functions.elementary.miscellaneous import sqrt, root

###############################################################################
################################ ERROR FUNCTION ###############################
###############################################################################

class erf(Function):
    """
    The Gauss error function.

    This function is defined as:

    :math:`\\mathrm{erf}(x)=\\frac{2}{\\sqrt{\\pi}} \\int_0^x e^{-t^2} \\, \\mathrm{d}x`

    Or, in ASCII::

                x
            /
           |
           |     2
           |   -t
        2* |  e    dt
           |
          /
          0
        -------------
              ____
            \/ pi


    """

    nargs = 1
    unbranched = True

    def fdiff(self, argindex=1):
        if argindex == 1:
            return 2*C.exp(-self.args[0]**2)/sqrt(S.Pi)
        else:
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, arg):
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return S.One
            elif arg is S.NegativeInfinity:
                return S.NegativeOne
            elif arg is S.Zero:
                return S.Zero
            elif arg.is_negative:
                return -cls(-arg)
        elif arg.could_extract_minus_sign():
            return -cls(-arg)

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)

            k = (n - 1)//2

            if len(previous_terms) > 2:
                return -previous_terms[-2] * x**2 * (n-2)/(n*k)
            else:
                return 2*(-1)**k * x**n/(n*C.factorial(k)*sqrt(S.Pi))

    def _eval_as_leading_term(self, x):
        arg = self.args[0].as_leading_term(x)

        if C.Order(1,x).contains(arg):
            return arg
        else:
            return self.func(arg)

    def _eval_is_real(self):
        return self.args[0].is_real

###############################################################################
#################### FRESNEL INTEGRALS ########################################
###############################################################################

class FresnelIntegral(Function):
    """ Base class for the Fresnel integrals."""

    nargs = 1

    _trigfunc = None
    _sign = None

    @classmethod
    def eval(cls, z):
        # Value at zero
        if z is S.Zero:
            return S(0)

        # Try to pull out factors of -1 and I
        prefact = S.One
        newarg = z
        changed = False

        nz = newarg.extract_multiplicatively(-1)
        if nz is not None:
            prefact = -prefact
            newarg = nz
            changed = True

        nz = newarg.extract_multiplicatively(I)
        if nz is not None:
            prefact = cls._sign*I*prefact
            newarg = nz
            changed = True

        if changed:
            return prefact*cls(newarg)

        # Values at infinities
        if z is S.Infinity:
            return S.Half
        #elif z is S.NegativeInfinity:
        #    return -S.Half
        elif z is I*S.Infinity:
            return cls._sign*I*S.Half
        #elif z is I*S.NegativeInfinity:
        #    return -cls._sign*I*S.Half

    def fdiff(self, argindex=1):
        if argindex == 1:
            return self._trigfunc(S.Half*pi*self.args[0]**2)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate())

    def _as_real_imag(self, deep=True, **hints):
        if self.args[0].is_real:
            if deep:
                hints['complex'] = False
                return (self.expand(deep, **hints), S.Zero)
            else:
                return (self, S.Zero)
        if deep:
            re, im = self.args[0].expand(deep, **hints).as_real_imag()
        else:
            re, im = self.args[0].as_real_imag()
        return (re, im)

    def as_real_imag(self, deep=True, **hints):
        x, y = self._as_real_imag(deep=deep, **hints)
        sq = -y**2/x**2
        re = S.Half*(self.func(x+x*sqrt(sq))+self.func(x-x*sqrt(sq)))
        im = x/(2*y) * sqrt(sq) * (self.func(x-x*sqrt(sq)) - self.func(x+x*sqrt(sq)))
        return (re, im)

    def _eval_expand_complex(self, deep=True, **hints):
        re_part, im_part = self.as_real_imag(deep=deep, **hints)
        return re_part + im_part*S.ImaginaryUnit


class fresnel_S(FresnelIntegral):
    r"""
    Fresnel integral S.

    This function is defined by

    .. math:: \operatorname{S}(z) = \int_0^z \sin{\frac{\pi}{2} t^2} \mathrm{d}t.

    It is an entire function.
    """

    _trigfunc = C.sin
    _sign = -S.One

    def _eval_rewrite_as_erf(self, z):
        return (S.One+I)/4 * (erf((S.One+I)/2*sqrt(pi)*z) - I*erf((S.One-I)/2*sqrt(pi)*z))

    def _eval_aseries(self, n, args0, x, logx):
        z = self.args[0]
        e = S.Half*I*pi*z**2
        h1 = C.hyper([S.One,S.Half],[],2*I/(pi*z**2))
        h2 = C.hyper([S.One,S.Half],[],-2*I/(pi*z**2))
        return root(z**4,4)/(2*z) - S.One/(2*pi*z)*(C.exp(-e)*h1 + C.exp(e)*h2)


class fresnel_C(FresnelIntegral):
    r"""
    Fresnel integral C.

    This function is defined by

    .. math:: \operatorname{C}(z) = \int_0^z \cos{\frac{\pi}{2} t^2} \mathrm{d}t.

    It is an entire function.
    """

    _trigfunc = C.cos
    _sign = S.One

    def _eval_rewrite_as_erf(self, z):
        return (S.One-I)/4 * (erf((S.One+I)/2*sqrt(pi)*z) + I*erf((S.One-I)/2*sqrt(pi)*z))

    def _eval_aseries(self, n, args0, x, logx):
        z = self.args[0]
        e = S.Half*I*pi*z**2
        h1 = C.hyper([S.One,S.Half],[],2*I/(pi*z**2))
        h2 = C.hyper([S.One,S.Half],[],-2*I/(pi*z**2))
        return (z**4)**C.Rational(3,4)/(2*z**3) + I/(2*pi*z)*(C.exp(-e)*h1 - C.exp(e)*h2)
