from sympy.core.cache import cacheit
from sympy.assumptions import Assume, get_local_assumptions, Q
from sympy.core import Symbol

def test_cacheit_doc():
    @cacheit
    def testfn():
        "test docstring"
        pass

    assert testfn.__doc__ == "test docstring"
    assert testfn.__name__ == "testfn"

cnt = 0
def test_assump():
    @cacheit
    def testfn(x):
        global cnt
        cnt += 1
    x = Symbol('x')
    assert cnt == 0
    testfn(x)
    assert cnt == 1
    testfn(x)
    assert cnt == 1
    get_local_assumptions().add(Assume(x, Q.positive))
    testfn(x)
    assert cnt == 2
    testfn(x)
    assert cnt == 2
