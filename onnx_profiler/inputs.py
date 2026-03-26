import numpy as np
import onnx

from .constants import ONNX_DTYPE_TO_NUMPY


def parse_input_shapes(shape_strs):
    overrides = {}
    if not shape_strs:
        return overrides
    for s in shape_strs:
        if ":" not in s:
            raise ValueError(f"--input-shape の形式が不正です: '{s}' ('入力名:d0,d1,...' の形式で指定)")
        name, dims_str = s.split(":", 1)
        dims = [int(d) for d in dims_str.split(",")]
        overrides[name] = dims
    return overrides


def parse_input_values(value_strs):
    """'name=val0,val1,...' 形式で固定入力値を指定."""
    overrides = {}
    if not value_strs:
        return overrides
    for s in value_strs:
        if "=" not in s:
            raise ValueError(f"--input-value の形式が不正です: '{s}' ('入力名=v0,v1,...' の形式で指定)")
        name, vals_str = s.split("=", 1)
        vals = [float(v) for v in vals_str.split(",")]
        overrides[name] = vals
    return overrides


def resolve_shape(shape_proto, dynamic_size=1):
    dims = []
    for dim in shape_proto.dim:
        if dim.dim_value > 0:
            dims.append(dim.dim_value)
        else:
            dims.append(dynamic_size)
    return dims


def generate_random_inputs(model_path, dynamic_size=1, shape_overrides=None, value_overrides=None):
    model = onnx.load(model_path)
    graph = model.graph
    initializer_names = {init.name for init in graph.initializer}
    inputs = {}
    input_info = []
    for inp in graph.input:
        if inp.name in initializer_names:
            continue
        type_proto = inp.type.tensor_type
        elem_type = type_proto.elem_type
        np_dtype = ONNX_DTYPE_TO_NUMPY.get(elem_type)
        if np_dtype is None:
            print(f"警告: 入力 '{inp.name}' の型 {elem_type} は未対応です。float32 を使用します。")
            np_dtype = np.float32
        if shape_overrides and inp.name in shape_overrides:
            shape = shape_overrides[inp.name]
        else:
            dims = type_proto.shape.dim
            ndims = len(dims)
            num_dynamic = sum(1 for d in dims if d.dim_value <= 0)
            if ndims == 2 and num_dynamic == ndims:
                default_dyn = max(dynamic_size, 512)
            else:
                default_dyn = dynamic_size
            shape = resolve_shape(type_proto.shape, default_dyn)
        input_info.append({"name": inp.name, "shape": shape, "dtype": str(np_dtype)})
        if value_overrides and inp.name in value_overrides:
            vals = value_overrides[inp.name]
            data = np.array(vals, dtype=np_dtype).reshape(shape) if shape else np.array(vals[0], dtype=np_dtype)
            inputs[inp.name] = data
            continue
        if np.issubdtype(np_dtype, np.integer):
            data = np.random.randint(0, 256, size=shape).astype(np_dtype)
        elif np.issubdtype(np_dtype, np.floating):
            if len(shape) <= 1:
                data = np.array(np.random.uniform(256, 1024, size=shape if shape else []), dtype=np_dtype)
            else:
                data = np.array(np.random.rand(*shape), dtype=np_dtype)
        elif np_dtype == np.bool_:
            data = np.random.choice([True, False], size=shape)
        else:
            data = np.array(np.random.rand(*shape), dtype=np_dtype)
        inputs[inp.name] = data
    return inputs, input_info
