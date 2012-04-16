"""Classes for representing chain complexes."""

from sympy.polys.agca.homomorphisms import homomorphism

# TODO
# - manipulation of and operations on complexes (shift, sum, etc)
# - manipulation of free resolutions (lifting of maps, ...)

class ChainComplexFormat(object):
    """
    This objects stores information about how to print a chain complex.

    It is created by the ``format`` method of the chain_complex class.
    In contrast to other sympy objects, it is mutable.

    Attributes: (not all necessarily supported by all printers)

    - chaincomplex - the chain complex to print
    - start - the (index of the) first module to print
    - terms - the number modules to print
    - dotsleft - whether to print an ellipsis at the left
    - dotsright - whether to print an ellipsis at the right
    - differentials - whether to print the matrices of the differentials
    - degrees - whether to print the degree (index) of each module
    - name - if not None, print free modules as ``name**n``.

    Note: If terms is None, the printer is advised to make a sensible choice.
    One suggestion would be to print as many terms as fit on the screen, a
    maximum of ten, and stop after three consecutive zeros.
    """

    def __init__(self, cplx, start, terms, dotsleft, dotsright, differentials,
                 degrees, name):
        self.chaincomplex = cplx
        self.start = start
        self.terms = terms
        self.dotsleft = dotsleft
        self.dotsright = dotsright
        self.differentials = differentials
        self.degrees = degrees
        self.name = name

    def __repr__(self):
        return 'ChainComplexFormat(%s, start=%s, terms=%s, dotsleft=%s, ' \
               'dotsright=%s, differentials=%s, degrees=%s, name=%s)' % (
            self.chaincomplex, self.start, self.terms, self.dotsleft,
            self.dotsright, self.differentials, self.degrees, self.name)

class ChainComplex(object):
    r"""
    Base class for chain complexes.

    A chain complex `C` over a ring `R` is a sequence of `R`-modules `M_i`
    and `R`-module homomorphisms `d_i : M_i \to M_{i - 1}`, such that
    `d_i \ocirc d_{i+1} = 0`. Here `i` runs through all of `\mathbb{Z}`.

    The most important operation on chain complexes is forming homology:
    the `n`-th homology module of `C` is `ker(d_n)/im(d_{n+1})`.

    Attributes:

    - ring - base ring
    - _differentials - cache for differentials
    - _modules - cache for modules
    - _default_opts - default options for printing

    Non-implemented methods:

    - _d
    - _M
    """

    def __init__(self, ring):
        self.ring = ring
        self._differentials = {}
        self._modules = {}
        self._default_opts = {}

        self._default_opts['start'] = 0
        self._default_opts['terms'] = None
        self._default_opts['dotsleft'] = True
        self._default_opts['dotsright'] = True
        self._default_opts['differentials'] = True
        self._default_opts['degrees'] = True
        self._default_opts['name'] = None

    def _check_int(self, i):
        """Make sure ``i`` is equivalent to an ``int``, return it as ``int``."""
        i_ = int(i)
        if i != i_:
            raise ValueError("Index must be an integer, got %s" % i)
        return i_

    def _d(self, i):
        """Compute d_i. ``i`` is an int."""
        raise NotImplementedError

    def _M(self, i):
        """Compute M_i. ``i`` is an int."""
        raise NotImplementedError

    def d(self, i):
        r"""
        Compute the ``i``-th differential `d: M_i \to M_{i - 1}`.
        """
        i = self._check_int(i)
        if not i in self._differentials:
            self._differentials[i] = self._d(i)
        return self._differentials[i]

    def M(self, i):
        r"""
        Compute the ``i``-th module `M_i`.
        """
        i = self._check_int(i)
        if not i in self._modules:
            self._modules[i] = self._M(i)
        return self._modules[i]

    def H(self, i):
        r"""
        Compute the ``i``-th homology module `H_i = ker(d_i)/im(d_{i+1})`.
        """
        return self.d(i).kernel() / self.d(i + 1).image()

    def format(self, **opts):
        """
        Format ``self`` for printing.
        """
        for k, v in self._default_opts.iteritems():
            if not k in opts:
                opts[k] = v
        return ChainComplexFormat(self, **opts)

    def format_compact(self):
        """
        Format ``self`` for abbreviated printing.

        This method is called by printers to obtain default formatting.
        """
        return self.format(terms=3, differentials=False, degrees=False)

    def format_pager(self, **opts):
        """Format, then print using pager_print."""
        from sympy.printing.pretty import pager_print
        return pager_print(self.format(**opts))

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, self.ring)

class BoundedBelowComplex(ChainComplex):
    r"""
    Class for bounded below chain complexes.

    That is, chain complexes such that there exists `n_0 \in \mathbb{Z}` with
    `M_i = 0` for `i < n_0` (and then `d_i = 0` as well for `i < n_0`).

    This class is for practical use. As such, it accepts arguments
    ``M_generator``, ``d_generator`` which should be callables with the
    following syntax:

        M_generator(cplx, i)
        d_generator(cplx, i, M_i, M_{i-1})

    Attributes:

    - lowest_term - the number n_0
    - _M_generator
    - _d_generator
    """

    def __init__(self, ring, M_generator, d_generator, n0=0):
        ChainComplex.__init__(self, ring)
        self.lowest_term = self._check_int(n0)
        self._M_generator = M_generator
        self._d_generator = d_generator

        self._default_opts['start'] = self.lowest_term - 1
        self._default_opts['dotsleft'] = False

    def _M(self, i):
        if i < self.lowest_term:
            return self.ring.free_module(0)
        return self._M_generator(self, i)

    def _d(self, i):
        if i < self.lowest_term:
            return homomorphisms(self.M(i), self.M(i - 1), [])
        return self._d_generator(self, i, self.M(i), self.M(i - 1))

def ZeroConstantComplex(ring, module, n0=0):
    r"""
    Create a BoundedBelowComplex starting at ``n0``, where all modules are
    ``module`` and all homomorphisms are the zero morphism.
    """

    zm = [0]*module.rank
    M_gen = lambda a, b: module
    d_gen = lambda a, b, fro, to: homomorphism(fro, to, zm)
    return BoundedBelowComplex(ring, M_gen, d_gen, n0)

def RepeatedIsomorphismComplex(isom, n0=0):
    ring = isom.ring
    m1 = isom.codomain
    m2 = isom.domain
    def M_gen(cplx, i):
        return [m1, m2, ring.free_module(0)][(i - n0) % 3]
    def d_gen(cplx, i, fro, to):
        if fro == m2 and to == m1:
            return isom
        return homomorphism(fro, to, [0]*fro.rank)
    return BoundedBelowComplex(ring, M_gen, d_gen, n0)
