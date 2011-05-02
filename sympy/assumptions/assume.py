# doctests are disabled because of issue #1521
from sympy.logic.boolalg import Boolean, Not
import inspect

class AssumptionsContext(set):
    """Set representing assumptions.

    This is used to represent global assumptions, but you can also use this
    class to create your own local assumptions contexts. It is basically a thin
    wrapper to Python's set, so see its documentation for advanced usage.

    Examples:
        >>> from sympy import AssumptionsContext, Assume, Q
        >>> c = AssumptionsContext; c
        AssumptionsContext()
        >>> from sympy.abc import x
        >>> c.add(Assume(x, Q.real))
        >>> c
        AssumptionsContext([Assume(x, 'real', True)])
        >>> c.remove(Assume(x, Q.real))
        >>> c
        AssumptionsContext()
        >>> c.clear()

    """

    def __init__(self):
        self.inverse = {}

    def add(self, *assumptions):
        """Add an assumption."""
        for a in assumptions:
            assert type(a) is Assume or \
                   type(a) is Not, 'can only store instances of Assume'
            super(AssumptionsContext, self).add(a)
            for x in a.free_symbols:
                if x in self.inverse:
                    self.inverse[x].add(a)
                else:
                    self.inverse[x] = set([a])

    def clear_symbol(self, x):
        if not x in self.inverse:
            return
        for a in self.inverse[x]:
            self.remove(a)
        del self.inverse[x]

LOCALCONTEXT = '__sympy_local_assumptions'

# TODO these functions should explicitely del the frame

def set_local_assumptions():
    """Set a local assumption context and return it.

    By default, this is done in the current local scope, this can be changed
    using the `scope` argument.

    This will overwrite any previous (local) assumptions."""
    f = inspect.currentframe().f_back
    c = AssumptionsContext()
    f.f_locals[LOCALCONTEXT] = c
    return c

def get_local_assumptions():
    """Return the current local assumption context.

    This can be in an outer scope if `set_local_assumptions()` has not been yet
    called in the current scope.

    If there is no defined assumption context in any scope, None is returned.
    """
    f = inspect.currentframe()
    while f.f_back is not None:
        f = f.f_back
        result = f.f_locals.get(LOCALCONTEXT)
        if result is not None:
            return result
    return None

def push_local_assumptions(go_back=0):
    """Create a local assumptions context that inherits all
       parent assumptions.

       go_back can be used to inject assumptions in parent scope"""
    f = inspect.currentframe()

    for i in range(go_back):
        f = f.f_back
        r = f.f_locals.get(LOCALCONTEXT)
        if r is not None:
            raise RuntimeError('Local context found too early!')

    f = f.f_back
    here = f
    r = f.f_locals.get(LOCALCONTEXT)
    if r is None:
        r = AssumptionsContext()
    while f.f_back is not None:
        f = f.f_back
        ctx = f.f_locals.get(LOCALCONTEXT)
        if ctx is not None:
            r.add(*ctx)
    here.f_locals[LOCALCONTEXT] = r
    return r

class Assume(Boolean):
    """New-style assumptions.

    >>> from sympy import Assume, Q
    >>> from sympy.abc import x
    >>> Assume(x, Q.integer)
    Assume(x, Q.integer)
    >>> Assume(x, Q.integer, False)
    Not(Assume(x, Q.integer))
    >>> Assume( x > 1 )
    Assume(1 < x, Q.is_true)

    """
    def __new__(cls, expr, predicate=None, value=True):
        from sympy.assumptions.ask import Q
        if predicate is None:
            predicate = Q.is_true
        elif not isinstance(predicate, Predicate):
            key = str(predicate)
            try:
                predicate = getattr(Q, key)
            except AttributeError:
                predicate = Predicate(key)
        if value:
            return Boolean.__new__(cls, expr, predicate)
        else:
            return Not(Boolean.__new__(cls, expr, predicate))

    is_Atom = True # do not attempt to decompose this

    @property
    def expr(self):
        """
        Return the expression used by this assumption.

        Examples:
            >>> from sympy import Assume, Q
            >>> from sympy.abc import x
            >>> a = Assume(x+1, Q.integer)
            >>> a.expr
            1 + x

        """
        return self._args[0]

    @property
    def key(self):
        """
        Return the key used by this assumption.
        It is a string, e.g. 'integer', 'rational', etc.

        Examples:
            >>> from sympy import Assume, Q
            >>> from sympy.abc import x
            >>> a = Assume(x, Q.integer)
            >>> a.key
            Q.integer

        """
        return self._args[1]

    def __eq__(self, other):
        if type(other) == Assume:
            return self._args == other._args
        return False

    def __hash__(self):
        return super(Assume, self).__hash__()

def eliminate_assume(expr, symbol=None):
    """
    Convert an expression with assumptions to an equivalent with all assumptions
    replaced by symbols.

    Assume(x, integer=True) --> integer
    Assume(x, integer=False) --> ~integer

    Examples:
        >>> from sympy.assumptions.assume import eliminate_assume
        >>> from sympy import Assume, Q
        >>> from sympy.abc import x
        >>> eliminate_assume(Assume(x, Q.positive))
        Q.positive
        >>> eliminate_assume(Assume(x, Q.positive, False))
        Not(Q.positive)

    """
    if expr.func is Assume:
        if symbol is not None:
            if not expr.expr.has(symbol):
                return
        return expr.key

    if expr.func is Predicate:
        return expr

    # To be used when Python 2.4 compatability is no longer required
    #return expr.func(*[x for x in (eliminate_assume(arg, symbol) for arg in expr.args) if x is not None])
    return expr.func(*filter(lambda x: x is not None, [eliminate_assume(arg, symbol) for arg in expr.args]))

class Predicate(Boolean):

    is_Atom = True

    def __new__(cls, name, handlers=None):
        obj = Boolean.__new__(cls)
        obj.name = name
        obj.handlers = handlers or []
        return obj

    def _hashable_content(self):
        return (self.name,)

    def __getnewargs__(self):
        return (self.name,)

    def __call__(self, expr):
        return Assume(expr, self.name)

    def add_handler(self, handler):
        self.handlers.append(handler)

    def remove_handler(self, handler):
        self.handlers.remove(handler)
