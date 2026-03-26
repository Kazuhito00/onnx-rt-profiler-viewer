import numpy as np
import onnx

ONNX_DTYPE_TO_NUMPY = {
    onnx.TensorProto.FLOAT: np.float32,
    onnx.TensorProto.UINT8: np.uint8,
    onnx.TensorProto.INT8: np.int8,
    onnx.TensorProto.UINT16: np.uint16,
    onnx.TensorProto.INT16: np.int16,
    onnx.TensorProto.INT32: np.int32,
    onnx.TensorProto.INT64: np.int64,
    onnx.TensorProto.BOOL: np.bool_,
    onnx.TensorProto.FLOAT16: np.float16,
    onnx.TensorProto.DOUBLE: np.float64,
    onnx.TensorProto.UINT32: np.uint32,
    onnx.TensorProto.UINT64: np.uint64,
}

NUM_WARMUP = 1
NUM_RUNS = 3

OP_COLOR_PALETTE = [
    '#22c55e', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#14b8a6',
    '#f97316', '#06b6d4', '#d946ef', '#6366f1', '#0ea5e9', '#eab308',
    '#ec4899', '#10b981', '#a78bfa', '#64748b', '#be185d', '#0d9488',
    '#7c3aed', '#dc2626', '#059669', '#d97706', '#4f46e5', '#db2777',
]
