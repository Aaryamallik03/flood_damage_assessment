"""Retry a smaller model: prepared conv-only int8, plus an fp16-weights fallback."""
import sys, warnings
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod,
                                      QuantFormat, QuantType, quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process
from PIL import Image
from torchvision import transforms

warnings.filterwarnings("ignore")
CLASSES = ["no-damage", "minor-damage", "major-damage", "destroyed"]
out = Path("export")
fp32 = out / "damage_fp32.onnx"
if not fp32.exists():
    sys.exit("Run export_onnx.py first.")

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
pairs, labels = [], []
for idx, name in enumerate(CLASSES):
    for before in sorted((Path("calib") / name).glob("*_before.png")):
        after = before.with_name(before.name.replace("_before", "_after"))
        if after.exists():
            a = tf(Image.open(before).convert("RGB")).unsqueeze(0).numpy()
            b = tf(Image.open(after).convert("RGB")).unsqueeze(0).numpy()
            pairs.append((a, b))
            labels.append(idx)
labels = np.array(labels)
print(f"Loaded {len(pairs)} pairs.")


def evaluate(name, path, ref=None):
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    preds = np.array([int(sess.run(None, {"pre": a, "post": b})[0].argmax()) for a, b in pairs])
    msg = f"{name}: {Path(path).stat().st_size / 1e6:.1f} MB, accuracy {float((preds == labels).mean()):.3f}"
    if ref is not None:
        msg += f", same answer as fp32 on {float((preds == ref).mean()):.3f} of pairs"
    print(msg)
    return preds


ref = evaluate("fp32", fp32)


class Reader(CalibrationDataReader):
    def __init__(self, items):
        self.it = iter([{"pre": a, "post": b} for a, b in items])

    def get_next(self):
        return next(self.it, None)


# 1. Conv-only int8 after the preparation step
try:
    prepared = out / "damage_fp32_prepared.onnx"
    try:
        quant_pre_process(str(fp32), str(prepared))
    except Exception as exc:
        print("Preparation with symbolic shapes failed, retrying:", exc)
        quant_pre_process(str(fp32), str(prepared), skip_symbolic_shape=True)
    int8 = out / "damage_int8.onnx"
    quantize_static(
        str(prepared), str(int8), Reader(pairs[::2]),
        quant_format=QuantFormat.QOperator, per_channel=True,
        op_types_to_quantize=["Conv"],
        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.Percentile,
        extra_options={"CalibPercentile": 99.999},
    )
    evaluate("int8 (conv only)", int8, ref)
except Exception as exc:
    print("int8 attempt failed:", exc)

# 2. fp16 weights, fp32 compute
m = onnx.load(str(fp32))
keep, add, casts = [], [], []
for init in m.graph.initializer:
    if init.data_type == TensorProto.FLOAT and int(np.prod(init.dims)) >= 1024:
        arr = numpy_helper.to_array(init).astype(np.float16)
        half = numpy_helper.from_array(arr, init.name + "__fp16")
        add.append(half)
        casts.append(helper.make_node("Cast", [half.name], [init.name],
                                      to=TensorProto.FLOAT, name="cast_" + init.name))
    else:
        keep.append(init)
nodes = casts + list(m.graph.node)
del m.graph.initializer[:]
m.graph.initializer.extend(keep + add)
del m.graph.node[:]
m.graph.node.extend(nodes)
half_path = out / "damage_fp16w.onnx"
onnx.save(m, str(half_path))
evaluate("fp16 weights", half_path, ref)