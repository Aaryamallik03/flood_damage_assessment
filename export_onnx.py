"""Export the trained damage classifier to ONNX, then test fp32 and int8 versions."""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.models.damage_classifier import CLASS_NAMES, SiameseDamageClassifier

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", default="backend/models_store/damage_classifier.pt")
ap.add_argument("--calib", default="calib")
ap.add_argument("--out", default="export")
a = ap.parse_args()

out = Path(a.out)
out.mkdir(exist_ok=True)

model = SiameseDamageClassifier(backbone="resnet50", num_classes=4, pretrained=False)
model.load_state_dict(torch.load(a.checkpoint, map_location="cpu"))
model.eval()

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load calibration pairs: calib/<class name>/<id>_before.png and <id>_after.png
pairs, labels = [], []
for idx, name in enumerate(CLASS_NAMES):
    for before in sorted((Path(a.calib) / name).glob("*_before.png")):
        after = before.with_name(before.name.replace("_before", "_after"))
        if after.exists():
            pre = tf(Image.open(before).convert("RGB")).unsqueeze(0)
            post = tf(Image.open(after).convert("RGB")).unsqueeze(0)
            pairs.append((pre, post))
            labels.append(idx)
if not pairs:
    sys.exit(f"No image pairs found in {a.calib}. Run make_samples.py first.")
print(f"Loaded {len(pairs)} test pairs.")

# 1. Export fp32
fp32 = out / "damage_fp32.onnx"
dummy = torch.randn(1, 3, 224, 224)
kw = dict(input_names=["pre", "post"], output_names=["logits"])
try:
    torch.onnx.export(model, (dummy, dummy), str(fp32), opset_version=17, dynamo=False, **kw)
except Exception as exc:
    print("Legacy exporter failed, trying the default one:", exc)
    torch.onnx.export(model, (dummy, dummy), str(fp32), opset_version=18, **kw)
print("Saved", fp32, round(fp32.stat().st_size / 1e6, 1), "MB")

import onnxruntime as ort
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

def predict_torch(pre, post):
    with torch.no_grad():
        return model(pre, post).numpy()

def make_runner(path):
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return lambda pre, post: sess.run(None, {"pre": pre.numpy(), "post": post.numpy()})[0]

def evaluate(name, fn):
    preds = np.array([int(fn(p, q).argmax()) for p, q in pairs])
    acc = float((preds == np.array(labels)).mean())
    print(f"{name}: accuracy on these pairs = {acc:.3f}")
    return preds

torch_preds = evaluate("PyTorch", predict_torch)
run32 = make_runner(fp32)
onnx_preds = evaluate("ONNX fp32", run32)
diff = max(float(np.abs(predict_torch(p, q) - run32(p, q)).max()) for p, q in pairs[:10])
print(f"fp32 ONNX vs PyTorch: max logit difference = {diff:.5f}, same answer on "
      f"{float((torch_preds == onnx_preds).mean()):.3f} of pairs")

# 2. int8 version
class Reader(CalibrationDataReader):
    def __init__(self, items):
        self.items = iter([{"pre": p.numpy(), "post": q.numpy()} for p, q in items])
    def get_next(self):
        return next(self.items, None)

int8 = out / "damage_int8.onnx"
step = max(1, len(pairs) // 60)
quantize_static(
    str(fp32), str(int8), Reader(pairs[::step]),
    quant_format=QuantFormat.QDQ, per_channel=True,
    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
)
print("Saved", int8, round(int8.stat().st_size / 1e6, 1), "MB")
q_preds = evaluate("ONNX int8", make_runner(int8))
print(f"int8 gives the same answer as PyTorch on {float((torch_preds == q_preds).mean()):.3f} of pairs")