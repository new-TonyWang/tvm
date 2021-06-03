import tvm
import tvm.runtime.ndarray as array
import numpy as np
def test_a():
    c = array.numpyasarray(np.ones([2,2],dtype=float))
    a = array.array(np.ones([2,2],dtype=float))
    b = array.array(np.ones([2,2],dtype=float))

    npa = a.asnumpy();
    print( isinstance(str,type(npa.dtype)))
    print( npa.dtype)