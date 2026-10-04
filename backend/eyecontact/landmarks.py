"""MediaPipe's face-landmark network (478 points incl. irises), run on CUDA through torch.

MediaPipe itself has no GPU path on Windows, so the network is lifted out of its
`.task` bundle once and executed by a small torch interpreter. The model is a
plain CNN — conv / depthwise conv / PReLU / add / max-pool / pad — so the
interpreter is exact (checked against MediaPipe's own output to ~2px at 4K,
the difference being the crop each side fed it).
"""
import io
import logging
import math
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from config import DATA_DIR

logger = logging.getLogger(__name__)

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")
MODEL_DIR = DATA_DIR / "models"
TASK_PATH = MODEL_DIR / "face_landmarker.task"
SPEC_PATH = MODEL_DIR / "face_landmarks_detector.pt"
TFLITE_MEMBER = "face_landmarks_detector.tflite"
# The real bundle is ~3.7 MB; anything far smaller is an error page.
TASK_MIN_BYTES = 1_000_000


def ensure_model() -> Path:
    """Path to the converted network, downloading and converting it once."""
    if SPEC_PATH.exists():
        return SPEC_PATH
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if not TASK_PATH.exists() or TASK_PATH.stat().st_size < TASK_MIN_BYTES:
        logger.info("Downloading the face landmark model (once)")
        with urllib.request.urlopen(MODEL_URL, timeout=120) as resp:
            payload = resp.read()
        if len(payload) < TASK_MIN_BYTES:
            raise RuntimeError(f"Face landmark model download is not a model ({len(payload)} bytes)")
        TASK_PATH.write_bytes(payload)
    with zipfile.ZipFile(TASK_PATH) as bundle:
        tflite_bytes = bundle.read(TFLITE_MEMBER)
    spec = convert_tflite(tflite_bytes)
    tmp = SPEC_PATH.with_suffix(".tmp")
    torch.save(spec, tmp)
    tmp.replace(SPEC_PATH)
    logger.info("Converted the face landmark model to %s", SPEC_PATH)
    return SPEC_PATH


# --- TFLite -> plain dict ---------------------------------------------------

def convert_tflite(buf: bytes) -> dict:
    """Flatten a TFLite CNN into {ops, tensors, input, outputs}, weights in NCHW layout."""
    import tflite  # only needed for the one-off conversion

    opnames = {v: k for k, v in tflite.BuiltinOperator.__dict__.items() if isinstance(v, int)}
    act = {0: "none", 1: "relu", 3: "relu6"}
    pad_kind = {0: "same", 1: "valid"}
    dtypes = {0: np.float32, 1: np.float16, 2: np.int32, 3: np.uint8, 4: np.int64}

    m = tflite.Model.GetRootAsModel(buf, 0)
    g = m.Subgraphs(0)

    def const(idx):
        t = g.Tensors(idx)
        b = m.Buffers(t.Buffer())
        if b.DataLength() == 0:
            return None
        return np.frombuffer(b.DataAsNumpy().tobytes(), dtypes[t.Type()]).reshape(t.ShapeAsNumpy()).copy()

    consts, ops, alias = {}, [], {}

    def name(i):
        return alias.get(i, f"t{i}")

    for i in range(g.OperatorsLength()):
        op = g.Operators(i)
        oc = m.OperatorCodes(op.OpcodeIndex())
        kind = opnames[max(oc.BuiltinCode(), oc.DeprecatedBuiltinCode())]
        ins = [int(k) for k in op.InputsAsNumpy()]
        outs = [int(k) for k in op.OutputsAsNumpy()]
        if kind == "DEQUANTIZE":            # fp16 weights -> fp32 constants
            consts[f"c{outs[0]}"] = const(ins[0]).astype(np.float32)
            alias[outs[0]] = f"c{outs[0]}"
            continue
        for k in ins:
            if k >= 0 and k not in alias:
                arr = const(k)
                if arr is not None:
                    consts[f"c{k}"] = arr
                    alias[k] = f"c{k}"
        o = op.BuiltinOptions()
        rec = {"op": kind, "in": [name(k) for k in ins if k >= 0], "out": name(outs[0])}
        if kind in ("CONV_2D", "DEPTHWISE_CONV_2D"):
            opt = (tflite.Conv2DOptions if kind == "CONV_2D" else tflite.DepthwiseConv2DOptions)()
            opt.Init(o.Bytes, o.Pos)
            rec.update(stride=[opt.StrideH(), opt.StrideW()], pad=pad_kind[opt.Padding()],
                       act=act[opt.FusedActivationFunction()],
                       dil=[opt.DilationHFactor(), opt.DilationWFactor()])
        elif kind == "ADD":
            opt = tflite.AddOptions()
            opt.Init(o.Bytes, o.Pos)
            rec.update(act=act[opt.FusedActivationFunction()])
        elif kind == "MAX_POOL_2D":
            opt = tflite.Pool2DOptions()
            opt.Init(o.Bytes, o.Pos)
            rec.update(stride=[opt.StrideH(), opt.StrideW()], pad=pad_kind[opt.Padding()],
                       k=[opt.FilterHeight(), opt.FilterWidth()])
        elif kind == "RESHAPE":
            rec.update(shape=[int(x) for x in g.Tensors(outs[0]).ShapeAsNumpy()])
        elif kind not in ("PRELU", "PAD", "LOGISTIC"):
            raise NotImplementedError(f"TFLite op {kind} is not supported by the converter")
        ops.append(rec)

    tensors = {}
    for r in ops:
        for j, nm in enumerate(r["in"]):
            if not nm.startswith("c") or nm in tensors:
                continue
            a = consts[nm]
            if r["op"] == "CONV_2D" and j == 1:              # OHWI -> OIHW
                a = a.transpose(0, 3, 1, 2)
            elif r["op"] == "DEPTHWISE_CONV_2D" and j == 1:  # 1HWO -> O1HW
                a = a.transpose(3, 0, 1, 2)
            elif r["op"] == "PRELU":
                a = a.reshape(-1)[:, None, None]
            elif r["op"] == "ADD":
                a = a.reshape(-1)[:, None, None] if a.size > 1 else a.reshape(())
            elif r["op"] == "PAD":
                a = a.astype(np.int64)
            tensors[nm] = torch.from_numpy(np.ascontiguousarray(a))
    return {"ops": ops, "tensors": tensors, "input": f"t{g.Inputs(0)}",
            "outputs": [name(g.Outputs(k)) for k in range(g.OutputsLength())]}


# --- the interpreter -----------------------------------------------------------

def _same_pad(size, k, s, d=1):
    total = max((math.ceil(size / s) - 1) * s + (k - 1) * d + 1 - size, 0)
    return total // 2, total - total // 2


def _act(x, a):
    if a == "relu":
        return F.relu(x)
    if a == "relu6":
        return F.relu6(x)
    return x


class TFLiteNet(torch.nn.Module):
    """Runs a converted graph in NCHW. Outputs: [landmarks (N,1434), presence logit (N,1), ...]."""

    def __init__(self, spec):
        super().__init__()
        self.ops = [dict(r) for r in spec["ops"]]
        for r in self.ops:   # pad sizes are static: keep them host-side so CUDA graphs can capture
            if r["op"] == "PAD":
                r["pads"] = spec["tensors"][r["in"][1]].tolist()
        self.input = spec["input"]
        self.outputs = spec["outputs"]
        for k, v in spec["tensors"].items():
            self.register_buffer(k, v)

    def forward(self, x):
        env = {self.input: x}

        def get(n):
            return env[n] if n in env else getattr(self, n)

        for r in self.ops:
            op = r["op"]
            a = get(r["in"][0])
            if op in ("CONV_2D", "DEPTHWISE_CONV_2D"):
                w = get(r["in"][1])
                b = get(r["in"][2]) if len(r["in"]) > 2 else None
                (sh, sw), (dh, dw) = r["stride"], r["dil"]
                pad = 0
                if r["pad"] == "same":
                    ph = _same_pad(a.shape[2], w.shape[2], sh, dh)
                    pw = _same_pad(a.shape[3], w.shape[3], sw, dw)
                    if ph[0] != ph[1] or pw[0] != pw[1]:   # TF pads the far side more
                        a = F.pad(a, (pw[0], pw[1], ph[0], ph[1]))
                    else:
                        pad = (ph[0], pw[0])
                groups = a.shape[1] if op == "DEPTHWISE_CONV_2D" else 1
                y = _act(F.conv2d(a, w, b, (sh, sw), pad, (dh, dw), groups), r["act"])
            elif op == "PRELU":
                y = torch.where(a >= 0, a, a * get(r["in"][1]))
            elif op == "ADD":
                y = _act(a + get(r["in"][1]), r["act"])
            elif op == "MAX_POOL_2D":
                (kh, kw), (sh, sw) = r["k"], r["stride"]
                if r["pad"] == "same":
                    ph, pw = _same_pad(a.shape[2], kh, sh), _same_pad(a.shape[3], kw, sw)
                    a = F.pad(a, (pw[0], pw[1], ph[0], ph[1]), value=float("-inf"))
                y = F.max_pool2d(a, (kh, kw), (sh, sw))
            elif op == "PAD":
                p = r["pads"]   # NHWC [[n0,n1],[h0,h1],[w0,w1],[c0,c1]]
                y = F.pad(a, (p[2][0], p[2][1], p[1][0], p[1][1], p[3][0], p[3][1]))
            elif op == "RESHAPE":
                y = a.permute(0, 2, 3, 1).reshape(a.shape[0], *r["shape"][1:]) if a.dim() == 4 \
                    else a.reshape(a.shape[0], *r["shape"][1:])
            elif op == "LOGISTIC":
                y = torch.sigmoid(a)
            else:
                raise NotImplementedError(op)
            env[r["out"]] = y
        res = []
        for n in self.outputs:
            t = env[n]
            res.append(t.permute(0, 2, 3, 1).reshape(t.shape[0], -1) if t.dim() == 4 else t.reshape(t.shape[0], -1))
        return res


class GraphNet:
    """Fixed-batch landmark net replayed through a CUDA graph: ~220 tiny kernels become one launch."""

    def __init__(self, spec_path: Path, batch: int, device):
        self.net = TFLiteNet(torch.load(spec_path, weights_only=True)).to(device).eval()
        self.x = torch.zeros(batch, 3, 256, 256, device=device)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side), torch.no_grad():
            for _ in range(3):
                self.net(self.x)
        torch.cuda.current_stream().wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.no_grad():
            self.out = self.net(self.x)

    def __call__(self, rgb):
        self.x.copy_(rgb)
        self.graph.replay()
        return self.out
