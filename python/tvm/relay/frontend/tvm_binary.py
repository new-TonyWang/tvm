import numpy as np
import tvm._ffi
from tvm._ffi.base import _LIB, check_call, c_array, string_types, _FFI_MODE
from tvm._ffi.runtime_ctypes import DataType, TVMContext, TVMArray, TVMArrayHandle
from tvm._ffi.runtime_ctypes import DataTypeCode, tvm_shape_index_t
from . import _ffi_api
import ctypes
def _load_lib():
    """Load libary by searching possible path."""
    lib_path = libinfo.find_lib_path()
    # The dll search path need to be added explicitly in
    # windows after python 3.8
    if sys.platform.startswith("win32") and sys.version_info >= (3, 8):
        for path in libinfo.get_dll_directories():
            os.add_dll_directory(path)
    lib = ctypes.CDLL(lib_path[0], ctypes.RTLD_GLOBAL)
    lib.TVMGetLastError.restype = ctypes.c_char_p
    return lib, os.path.basename(lib_path[0])
def list_global_func_names():
    """Get list of global functions registered.

    Returns
    -------
    names : list
       List of global functions names.
    """
    plist = ctypes.POINTER(ctypes.c_char_p)()
    size = ctypes.c_uint()

    check_call(_LIB.TVMFuncListGlobalNames(ctypes.byref(size), ctypes.byref(plist)))
    fnames = []
    for i in range(size.value):
        fnames.append(py_str(plist[i]))
    return fnames
def NDarray_from_binary():
    pshape = ctypes.POINTER(ctypes.c_uint64);#shape
    pdtype = ctypes.c_char
    pndim = ctypes.c_int64
    pctx = ctypes.c_int64
    arr= _LIB.TVMInitNDarrayFromBin(ctypes.byref(pshape),ctypes.byref(pdtype),ctypes.byref(pndim),ctypes.byref(pctx))
    
    return 


    

def from_tvm_binary():
    json_graph = "";
    params = {"p0":123};
    return json_graph, params