import importlib
import logging
import os
import tempfile

import torch
from .common import device_from_inputs, fake_tensor_unsupported
from .registry import register_backend

try:
    import numpy as np

    _np_dtype = {
        torch.float16: np.float16,
        torch.float32: np.float32,
        torch.float64: np.float64,
        torch.uint8: np.uint8,
        torch.int8: np.int8,
        torch.int16: np.int16,
        torch.int32: np.int32,
        torch.int64: np.longlong,
        torch.bool: np.bool_,
    }

except ImportError:
    _np_dtype = None


log = logging.getLogger(__name__)


def default_provider(device_type):
    if "ONNXRT_PROVIDER" in os.environ:
        return os.environ["ONNXRT_PROVIDER"]
    return {
        "cpu": "CPUExecutionProvider",
        "cuda": "CUDAExecutionProvider",
        # "TensorrtExecutionProvider" is another option
    }[device_type]


def has_onnxruntime():
    try:
        importlib.import_module("onnxruntime")
        return True
    except ImportError:
        return False


@register_backend
@fake_tensor_unsupported
def onnxrt(gm, example_inputs, *, filename=None, provider=None):
    if filename is None:
        # 产生一个 file，然后递归调用 onnxrt
        with tempfile.NamedTemporaryFile(suffix=".onnx") as tmp:
            return onnxrt(gm, example_inputs, filename=tmp.name)

    import onnxruntime  # type: ignore[import]

    assert _np_dtype, "requires numpy"

    # 从输入中的第一个 tensor 获取设备类型
    device_type = device_from_inputs(example_inputs).type
    # 实际执行一下 torch 的 gm 获取输出 tensor
    example_outputs = gm(*example_inputs)
    # 如果没有输出就返回原 eager module
    if len(example_outputs) == 0:
        log.warning("Explicitly fall back to eager due to zero output")
        return gm.forward
    # 获取输出的规格
    output_spec = [
        (o.shape, o.dtype, o.layout, o.device, o.requires_grad) for o in example_outputs
    ]
    # 创建 input/output 的名字
    input_names = [f"i{i}" for i in range(len(example_inputs))]
    output_names = [f"o{x}" for x in range(len(example_outputs))]

    # 把 model 导出到了 onnx 格式，存到为文件
    # 参考：https://pytorch.org/docs/stable/onnx.html#torch.onnx.export
    torch.onnx.export(
        torch.jit.script(gm),
        example_inputs,
        filename,
        input_names=input_names,
        output_names=output_names,
    )
    del example_inputs, example_outputs

    # 基于 onnx 文件来提供一个 onnxruntime 推理 session
    if provider is None:
        provider = default_provider(device_type)
    assert provider in onnxruntime.get_available_providers()
    session = onnxruntime.InferenceSession(filename, providers=[provider])

    def _call(*initial_args):
        # 提供了一个 binding 来共享输入的内存，避免内存拷贝
        binding = session.io_binding()
        active_inputs = {inp.name for inp in session.get_inputs()}
        args = [a.contiguous() for a in initial_args]
        for name, value in zip(input_names, args):
            if name not in active_inputs:
                log.warning(
                    "input %s skipped as not found in onnx inference session", name
                )
                continue
            dev = value.device
            binding.bind_input(
                name,
                dev.type,
                dev.index or 0,
                _np_dtype[value.dtype],
                value.size(),
                value.data_ptr(),
            )
        # 创建输出 tensor
        outputs = [
            torch.empty(
                shape,
                dtype=dtype,
                layout=layout,
                device=device,
                requires_grad=requires_grad,
            )
            for shape, dtype, layout, device, requires_grad in output_spec
        ]

        # 构建输出 tensor 到输出 binding的映射
        for name, value in zip(output_names, outputs):
            dev = value.device
            binding.bind_output(
                name,
                dev.type,
                dev.index or 0,
                _np_dtype[value.dtype],
                value.size(),
                value.data_ptr(),
            )
        # 输入 binding 执行推理
        session.run_with_iobinding(binding)
        if device_type == "cpu":
            binding.copy_outputs_to_cpu()
        # 返回推理结果
        return outputs

    return _call
