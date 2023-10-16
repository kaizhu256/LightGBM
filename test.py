# !! https://lightgbm.readthedocs.io/en/latest/C-API.html

import ctypes
from platform import system

import numpy as np

if system() in ("Darwin"):
    lib_file = "./lib_lightgbm.dylib"
elif system() in ("Windows", "Microsoft"):
    lib_file = "./lib_lightgbm.dll"
else:
    lib_file = "./lib_lightgbm.so"
print(lib_file)
LIB = ctypes.cdll.LoadLibrary(lib_file)

LIB.LGBM_GetLastError.restype = ctypes.c_char_p

dtype_float32 = 0
dtype_float64 = 1
dtype_int32 = 2
dtype_int64 = 3

def _safe_call(ret: int) -> None:
    """
    Check the return value from C API call.

    Parameters
    ----------
    ret : int
        The return value from C API calls.
    """
    if ret != 0:
        raise Exception(LIB.LGBM_GetLastError().decode("utf-8"))


def c_str(string: str) -> ctypes.c_char_p:
    """Convert a Python string to C string."""
    return ctypes.c_char_p(string.encode("utf-8"))


def save_to_binary(handle, filename):
    _safe_call(LIB.LGBM_DatasetSaveBinary(handle, c_str(filename)))


"""
def load_from_file(filename, reference):
    ref = None
    if reference is not None:
        ref = reference
    handle = ctypes.c_void_p()
    _safe_call(LIB.LGBM_DatasetCreateFromFile(
        c_str(str(filename)),
        c_str("max_bin=15"),
        ref,
        ctypes.byref(handle)))
    print(LIB.LGBM_GetLastError())
    num_data = ctypes.c_int(0)
    _safe_call(LIB.LGBM_DatasetGetNumData(handle, ctypes.byref(num_data)))
    num_feature = ctypes.c_int(0)
    _safe_call(LIB.LGBM_DatasetGetNumFeature(handle, ctypes.byref(num_feature)))
    print(f"#data: {num_data.value} #feature: {num_feature.value}")
    return handle


def load_from_csr(filename, reference):
    data = np.loadtxt(str(filename), dtype=np.float64)
    csr = sparse.csr_matrix(data[:, 1:])
    label = data[:, 0].astype(np.float32)
    handle = ctypes.c_void_p()
    ref = None
    if reference is not None:
        ref = reference

    _safe_call(LIB.LGBM_DatasetCreateFromCSR(
        csr.indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_int(dtype_int32),
        csr.indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        csr.data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(dtype_float64),
        ctypes.c_int64(len(csr.indptr)),
        ctypes.c_int64(len(csr.data)),
        ctypes.c_int64(csr.shape[1]),
        c_str("max_bin=15"),
        ref,
        ctypes.byref(handle)))
    num_data = ctypes.c_int(0)
    _safe_call(LIB.LGBM_DatasetGetNumData(handle, ctypes.byref(num_data)))
    num_feature = ctypes.c_int(0)
    _safe_call(LIB.LGBM_DatasetGetNumFeature(handle, ctypes.byref(num_feature)))
    _safe_call(LIB.LGBM_DatasetSetField(
        handle,
        c_str("label"),
        label.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(len(label)),
        ctypes.c_int(dtype_float32)))
    print(f"#data: {num_data.value} #feature: {num_feature.value}")
    return handle


def load_from_csc(filename, reference):
    data = np.loadtxt(str(filename), dtype=np.float64)
    csc = sparse.csc_matrix(data[:, 1:])
    label = data[:, 0].astype(np.float32)
    handle = ctypes.c_void_p()
    ref = None
    if reference is not None:
        ref = reference

    _safe_call(LIB.LGBM_DatasetCreateFromCSC(
        csc.indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_int(dtype_int32),
        csc.indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        csc.data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(dtype_float64),
        ctypes.c_int64(len(csc.indptr)),
        ctypes.c_int64(len(csc.data)),
        ctypes.c_int64(csc.shape[0]),
        c_str("max_bin=15"),
        ref,
        ctypes.byref(handle)))
    num_data = ctypes.c_int(0)
    _safe_call(LIB.LGBM_DatasetGetNumData(handle, ctypes.byref(num_data)))
    num_feature = ctypes.c_int(0)
    _safe_call(LIB.LGBM_DatasetGetNumFeature(handle, ctypes.byref(num_feature)))
    _safe_call(LIB.LGBM_DatasetSetField(
        handle,
        c_str("label"),
        label.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(len(label)),
        ctypes.c_int(dtype_float32)))
    print(f"#data: {num_data.value} #feature: {num_feature.value}")
    return handle


def free_dataset(handle):
    _safe_call(LIB.LGBM_DatasetFree(handle))


def test_dataset():
    train = load_from_file("binary.train", None)
    test = load_from_mat("binary.test", train)
    free_dataset(test)
    test = load_from_csr("binary.test", train)
    free_dataset(test)
    test = load_from_csc("binary.test", train)
    free_dataset(test)
    save_to_binary(train, "train.binary.bin")
    free_dataset(train)
    train = load_from_file("train.binary.bin", None)
    free_dataset(train)
"""


def load_from_mat(filename, reference):
    mat = np.loadtxt(str(filename), dtype=np.float64)
    label = mat[:, 0].astype(np.float32)
    mat = mat[:, 1:]
    data = np.array(mat.reshape(mat.size), dtype=np.float64, copy=False)
    handle = ctypes.c_void_p()
    ref = None
    if reference is not None:
        ref = reference

    _safe_call(LIB.LGBM_DatasetCreateFromMat(
        data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(dtype_float64),
        ctypes.c_int32(mat.shape[0]),
        ctypes.c_int32(mat.shape[1]),
        ctypes.c_int(1),
        c_str("max_bin=15"),
        ref,
        ctypes.byref(handle)))
    num_data = ctypes.c_int(0)
    _safe_call(LIB.LGBM_DatasetGetNumData(handle, ctypes.byref(num_data)))
    num_feature = ctypes.c_int(0)
    _safe_call(LIB.LGBM_DatasetGetNumFeature(handle, ctypes.byref(num_feature)))
    _safe_call(LIB.LGBM_DatasetSetField(
        handle,
        c_str("label"),
        label.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(len(label)),
        ctypes.c_int(dtype_float32)))
    print(f"#data: {num_data.value} #feature: {num_feature.value}")
    return handle


def test_booster():
    train = load_from_mat("binary.train", None)
    test = load_from_mat("binary.test", train)
    booster = ctypes.c_void_p()
    _safe_call(LIB.LGBM_BoosterCreate(
        train,
        c_str("app=binary metric=auc num_leaves=31 verbose=0"),
        ctypes.byref(booster)))
    _safe_call(LIB.LGBM_BoosterAddValidData(booster, test))
    is_finished = ctypes.c_int(0)
    for i in range(1, 51):
        _safe_call(LIB.LGBM_BoosterUpdateOneIter(booster, ctypes.byref(is_finished)))
        result = np.array([0.0], dtype=np.float64)
        out_len = ctypes.c_int(0)
        _safe_call(LIB.LGBM_BoosterGetEval(
            booster,
            ctypes.c_int(0),
            ctypes.byref(out_len),
            result.ctypes.data_as(ctypes.POINTER(ctypes.c_double))))
        if i % 10 == 0:
            print(f"{i} iteration test AUC {result[0]:.6f}")
    _safe_call(LIB.LGBM_BoosterSaveModel(
        # 00. BoosterHandle handle,
        booster,
        # 01. int start_iteration,
        ctypes.c_int(0),
        # 02. int num_iteration,
        ctypes.c_int(-1),
        # 03. int feature_importance_type,
        ctypes.c_int(0),
        # 04. const char *filename
        c_str("model.txt")))
    # !! LIGHTGBM_C_EXPORT int LGBM_BoosterSaveModel(
        # !! BoosterHandle handle,
        # !! int start_iteration,
        # !! int num_iteration,
        # !! int feature_importance_type,
        # !! const char *filename
    # !! )

    buffer_len = 0x100000
    tmp_out_len = ctypes.c_int64(0)
    string_buffer = ctypes.create_string_buffer(buffer_len)
    ptr_string_buffer = ctypes.c_char_p(*[ctypes.addressof(string_buffer)])
    # LIGHTGBM_C_EXPORT int LGBM_BoosterSaveModelToString(
    #     BoosterHandle handle,
    #     int start_iteration,
    #     int num_iteration,
    #     int feature_importance_type,
    #     int64_t buffer_len,
    #     int64_t *out_len, char *out_str
    # )
    _safe_call(LIB.LGBM_BoosterSaveModelToString(
        booster,
        ctypes.c_int(0),
        ctypes.c_int(-1),
        ctypes.c_int(0),
        ctypes.c_int64(buffer_len),
        ctypes.byref(tmp_out_len),
        ptr_string_buffer))
    actual_len = tmp_out_len.value
    # if buffer length is not long enough, re-allocate a buffer
    if actual_len > buffer_len:
        string_buffer = ctypes.create_string_buffer(actual_len)
        ptr_string_buffer = ctypes.c_char_p(*[ctypes.addressof(string_buffer)])
        _safe_call(LIB.LGBM_BoosterSaveModelToString(
            booster,
            ctypes.c_int(0),
            ctypes.c_int(-1),
            ctypes.c_int(0),
            ctypes.c_int64(actual_len),
            ctypes.byref(tmp_out_len),
            ptr_string_buffer))
    _safe_call(LIB.LGBM_BoosterFree(booster))
    _safe_call(LIB.LGBM_DatasetFree(train))
    _safe_call(LIB.LGBM_DatasetFree(test))
    booster2 = ctypes.c_void_p()
    model_str = string_buffer.value.decode("utf-8")

    # LIGHTGBM_C_EXPORT int LGBM_BoosterLoadModelFromString(
    #     const char *model_str,
    #     int *out_num_iterations,
    #     BoosterHandle *out
    # )
    _safe_call(LIB.LGBM_BoosterLoadModelFromString(
        c_str(model_str),
        ctypes.byref(ctypes.c_int(0)),
        ctypes.byref(booster2)))
    data = np.loadtxt("binary.test", dtype=np.float64)
    mat = data[:, 1:]
    preb = np.empty(mat.shape[0], dtype=np.float64)
    num_preb = ctypes.c_int64(0)
    data = np.array(mat.reshape(mat.size), dtype=np.float64, copy=False)

    # !! LIGHTGBM_C_EXPORT int LGBM_BoosterPredictForMat(
        # !! BoosterHandle handle,
        # !! const void *data,
        # !! int data_type,
        # !! int32_t nrow,
        # !! int32_t ncol,
        # !! int is_row_major,
        # !! int predict_type,
        # !! int start_iteration,
        # !! int num_iteration,
        # !! const char *parameter,
        # !! int64_t *out_len,
        # !! double *out_result)
    _safe_call(LIB.LGBM_BoosterPredictForMat(
        # 00. BoosterHandle handle
        booster2,
        # 01. const void *data
        data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        # 02. int data_type
        ctypes.c_int(dtype_float64),
        # 03. int32_t nrow
        ctypes.c_int32(mat.shape[0]),
        # 04. int32_t ncol
        ctypes.c_int32(mat.shape[1]),
        # 05. int is_row_major
        ctypes.c_int(1),
        # 06. int predict_type
        ctypes.c_int(1),
        # 07. int start_iteration
        ctypes.c_int(0),
        # 08. int num_iteration
        ctypes.c_int(25),
        # 09. const char *parameter
        c_str(""),
        # 10. int64_t *out_len
        ctypes.byref(num_preb),
        # 11. double *out_result
        preb.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    ))
    # !! print(preb)
    _safe_call(LIB.LGBM_BoosterPredictForFile(
        booster2,
        c_str("binary.test"),
        ctypes.c_int(0),
        ctypes.c_int(0),
        ctypes.c_int(0),
        ctypes.c_int(25),
        c_str(""),
        c_str("preb.txt")))
    _safe_call(LIB.LGBM_BoosterPredictForFile(
        booster2,
        c_str("binary.test"),
        ctypes.c_int(0),
        ctypes.c_int(0),
        ctypes.c_int(10),
        ctypes.c_int(25),
        c_str(""),
        c_str("preb.txt")))
    _safe_call(LIB.LGBM_BoosterFree(booster2))
# !! test_dataset()
test_booster()
