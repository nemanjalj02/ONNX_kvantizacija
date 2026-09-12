#zajednicki alati koji se koriste u skriptama za kvantizaciju
import numpy as np
from onnx import helper, numpy_helper

#kvantizacija po kanalu, koristi se za tezine
def quantize_per_channel(tensor: np.ndarray, qmax: int):

    axes = tuple(range(1, tensor.ndim))
    max_abs = np.max(np.abs(tensor), axis=axes)
    max_abs[max_abs == 0] = 1.0 #slucaj kanala gde su sve tezine nula
    scale = max_abs / qmax

    broadcast_shape = (-1,) + (1,) * (tensor.ndim - 1)
    q = np.round(tensor / scale.reshape(broadcast_shape))
    q = np.clip(q, -qmax, qmax)
    return q, scale

#kvantizacija po tenzoru, koristi se za biase
def quantize_per_tensor(tensor: np.ndarray, qmax: int):
    
    max_abs = np.abs(tensor).max()
    max_abs = 1.0 if max_abs == 0 else max_abs
    scale = np.float32(max_abs / qmax)

    q = np.round(tensor / scale)
    q = np.clip(q, -qmax, qmax)
    return q, scale

#umetanje dequantize cvorova u graf
def replace_with_dequantized(graph, name: str, q: np.ndarray, scale, np_dtype, axis=None):
    #nalazi originalni incijalizator u fp32 grafu
    init = next(i for i in graph.initializer if i.name == name)

    q_name = f"{name}_quantized"
    scale_name = f"{name}_scale"
    zp_name = f"{name}_zero_point"
    dq_name = f"{name}_dequantized"

    scale_arr = np.asarray(scale, dtype=np.float32)
    #pravi tri nova onnx tenzora
    q_init = numpy_helper.from_array(q.astype(np_dtype), name=q_name)
    scale_init = numpy_helper.from_array(scale_arr, name=scale_name)
    zp_init = numpy_helper.from_array(np.zeros_like(scale_arr, dtype=np_dtype), name=zp_name)
    kwargs = {"axis": axis} if axis is not None else {}
    dq_node = helper.make_node(
        "DequantizeLinear",
        inputs=[q_name, scale_name, zp_name],
        outputs=[dq_name],
        name=f"{name}_dequantize",
        **kwargs,
    )
    #brise stari fp32 inicijalizator i ubacuje novi umesto njega
    graph.initializer.remove(init)
    graph.initializer.extend([q_init, scale_init, zp_init])

    consumer_idx = next(i for i, n in enumerate(graph.node) if name in n.input)
    graph.node.insert(consumer_idx, dq_node)

    for n in graph.node:
        for i, inp in enumerate(n.input):
            if inp == name:
                n.input[i] = dq_name
