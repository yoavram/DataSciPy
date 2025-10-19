import fibonacci as fb
import numpy as np

assert fb.fib(8) == 13

assert (fb.fibseries([5,13,2,6]) == np.array([  3, 144,   1,   5], dtype=np.int32)).all()

print("Test passed.")