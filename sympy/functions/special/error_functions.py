from sympy.core import S, C, sympify, cacheit
from sympy.core.function import Function, ArgumentIndexError
from sympy.functions.elementary.miscellaneous import sqrt

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

    def _eval_rewrite_as_tractable(self, z):
        return S.One - erfs(z)*C.exp(-z**2)


class erfs(Function):
    """
    Helper function to make the :math:`erf(z)` function
    tractable for the Gruntz algorithm.
    """

    def _eval_aseries(self, n, args0, x, logx):
        if args0[0] != oo:
            return super(erfs, self)._eval_aseries(n, args0, x, logx)

        z = self.args[0]
        l = [ C.factorial(2*k)*(-4)**(-k)/C.factorial(k)*(1/z)**(2*k+1) for k in xrange(1,n) ]

        # Not sure about the order terms
        o = None
        if m == 0:
            o = C.Order(1, x)
        else:
            o = C.Order(1/z**(2*n+2), x)
        # It is very inefficient to first add the order and then do the nseries
        return (Add(*l))._eval_nseries(x, n, logx) + o


    def fdiff(self, argindex=1):
        if argindex == 1:
            z = self.args[0]
            return -2/sqrt(S.Pi)+2*z*erfs(z)
        else:
            raise ArgumentIndexError(self, argindex)
