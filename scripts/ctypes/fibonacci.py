
"""
Filename: fibonacci.py
Demonstrates the use of ctypes with three functions:

    (1) fib(a)
    (2) fibseries(b)
    (3) fibmatrix(c)

See: https://scipy-cookbook.readthedocs.org/items/Ctypes.html#fibonacci-example-using-numpy-arrays-c-and-scons
     http://www.lejordet.com/2009/04/simple-python-ctypes-example-in-windows/
"""
import numpy as np
import ctypes as ct
import sys
if sys.platform == 'win32': # win
    lib_filename = 'fibonacci.dll'
elif sys.platform == 'darwin': # osx
    lib_filename = 'libfibonacci.dylib'
else: # linux
    lib_filename = 'libfibonacci.so'
# Load the library as _libfibonacci.
# Why the underscore (_) in front of _libfibonacci below?
# To minimise namespace pollution -- see PEP 8 (www.python.org).
_libfibonacci = np.ctypeslib.load_library(lib_filename, '.')

_libfibonacci.fib.argtypes = [ct.c_int] #  Declare arg type, same below.
_libfibonacci.fib.restype  =  ct.c_int  #  Declare result type, same below.

_libfibonacci.fibseries.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.intc),
    ct.c_int,
    np.ctypeslib.ndpointer(dtype=np.intc)
]
_libfibonacci.fibseries.restype = ct.c_void_p

_libfibonacci.fibmatrix.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.intc),
    ct.c_int, ct.c_int,
    np.ctypeslib.ndpointer(dtype=np.intc)
]
_libfibonacci.fibmatrix.restype = ct.c_void_p

def fib(a):
    """Compute the n'th Fibonacci number.

    Arguments
    ---------
    a : int

    Returns
    -------
    int
        The n'th Fibonacci number.

    Examples
    --------
    >>> fib(8)
    13
    >>> fib(23)
    17711
    >>> fib(0)
    -1
    """
    return _libfibonacci.fib(int(a))

def fibseries(b):
    """Compute an array containing the n'th Fibonacci number of each entry.

    Arguments
    ---------
    b : list or numpy.ndarray
        a sequence of integers

    Returns
    -------
    numpy.ndarray
        array containing the n'th Fibonacci number of each entry

    Examples
    --------
    >>> fibseries([1, 2, 3, 4, 5, 6, 7, 8])
    array([ 0,  1,  1,  2,  3,  5,  8, 13])
    >>> fibseries(range(1, 12))
    array([ 0,  1,  1,  2,  3,  5,  8, 13, 21, 34, 55])
    """
    b = np.asarray(b, dtype=np.intc)
    result = np.empty(len(b), dtype=np.intc)
    _libfibonacci.fibseries(b, len(b), result)
    return result

def fibmatrix(c):
    """Compute a matrix containing the n'th Fibonacci number of each entry.

    Arguments
    ---------
    c : list or numpy.ndarray
        A nested list or 2D NumPy array of integers

    Returns
    -------
        NumPy array containing the n'th Fibonacci number of each entry

    Examples
    --------
    >>> from numpy import array
    >>> fibmatrix([[3,4],[5,6]])
    array([[1, 2],
           [3, 5]])
    >>> fibmatrix(array([[1,2,3],[4,5,6],[7,8,9]]))
    array([[ 0,  1,  1],
           [ 2,  3,  5],
           [ 8, 13, 21]])
    """
    tmp = np.asarray(c)
    rows, cols = tmp.shape
    c = tmp.astype(np.intc)
    result = np.empty(c.shape, dtype=np.intc)
    _libfibonacci.fibmatrix(c, rows, cols, result)
    return result

