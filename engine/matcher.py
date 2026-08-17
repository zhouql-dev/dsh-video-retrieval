#!/usr/bin/env python3
"""Identity matcher backends for person_search.py.

Three optional on-device embedders, each import-guarded so the engine NEVER
crashes when one is missing — it degrades: embedding -> CLIP -> VLM.

  FACE_OK : insightface (SCRFD/RetinaFace detect + ArcFace 512-d embed), via onnxruntime.
  REID_OK : OSNet-x0_25 person-ReID (scripts/osnet_loader.py) IF a weight file is present.
  CLIP_OK : OpenAI CLIP ViT-B/32 (weak for identity; always-available body fallback).

Heavy models load lazily on first use and are cached. InsightFace `buffalo_l`
auto-downloads to ~/.insightface on first use. OSNet weights are NOT bundled —
set OSNET_WEIGHTS to a ReID checkpoint (see references/setup.md).
"""
from __future__ import annotations
import glob
import os
import numpy as np
import cv2

here = os.path.dirname(os.path.abspath(__file__))
import sys; sys.path.insert(0, here)
from common import cosine, pick_device, skill_root, models_dir  # noqa: E402


def _first_existing(candidates, default):
    """First existing path among candidates, else default (a writable target for
    setup.py / clip.load downloads)."""
    for c in candidates:
        c = os.path.expanduser(c)
        if os.path.exists(c):
            return c
    return os.path.expanduser(default)


# ---- availability flags (computed without loading heavy models) ----
FACE_OK = False
try:
    from insightface.app import FaceAnalysis  # noqa: F401
    FACE_OK = True
except Exception:
    FACE_OK = False

# CLIP weights: env override -> bundled skill weights -> user models dir -> legacy
# Mac path (back-compat). clip.load(download_root=...) also downloads here if absent.
_CLIP_ROOT = os.environ.get("CLIP_WEIGHTS") or _first_existing(
    [str(skill_root() / "weights" / "clip"),
     str(models_dir() / "clip"),
     "~/dev/video-retrieval/weights/clip"],
    default=str(skill_root() / "weights" / "clip"))
try:
    import clip  # noqa: F401
    # The ViT-B-32.pt on this host is the standard OpenAI JIT checkpoint. Direct
    # torch.jit.load + encode_image fails on CPU/MPS (hardcoded CUDA device in the
    # TorchScript), but clip.load() extracts the state_dict and builds a proper
    # device-agnostic model. CLIP is a weak identity matcher — prefer OSNet.
    CLIP_OK = True
except Exception:
    CLIP_OK = False

def _resolve_osnet():
    """Find an OSNet x0.25 checkpoint. The HF-mirror download keeps the FULL
    training-run suffix (osnet_x0_25_msmt17_combineall_..._jitter.pth) while
    the historical code expected the short name — that mismatch silently set
    REID_OK=False and degraded person search to CLIP for a long time. Match
    either, preferring an explicit OSNET_WEIGHTS env / short name, then glob."""
    if os.environ.get("OSNET_WEIGHTS"):
        return os.path.expanduser(os.environ["OSNET_WEIGHTS"])
    candidates = [
        str(skill_root() / "weights" / "osnet"),
        str(models_dir() / "osnet"),
        os.path.expanduser("~/dev/video-retrieval/weights/osnet"),
    ]
    for d in candidates:
        short = os.path.join(d, "osnet_x0_25_msmt17.pth")
        if os.path.exists(short):
            return short
        try:
            hits = sorted(glob.glob(os.path.join(d, "osnet_x0_25*.pth")))
        except Exception:
            hits = []
        if hits:
            return hits[0]
    return str(skill_root() / "weights" / "osnet" / "osnet_x0_25_msmt17.pth")


OSNET_WEIGHTS = _resolve_osnet()
REID_OK = os.path.exists(OSNET_WEIGHTS)

# thresholds
FACE_THRESH = 0.42
REID_THRESH = 0.55
CLIP_THRESH = 0.78

# cached singletons
_face_app = None
_OSNET = None
_OSNET_X1 = None
_CLIP = None


# ---------------- FACE (insightface) ----------------
def _default_onnx_providers():
    """Per-OS ONNX provider preference: CoreML on macOS, CUDA where onnxruntime-gpu
    exposes it (Windows/Linux NVIDIA), CPU always last. onnxruntime silently ignores
    providers it doesn't know on some versions, but we filter to the available set
    to avoid noisy warnings."""
    import platform
    prov = []
    if platform.system() == "Darwin":
        prov.append("CoreMLExecutionProvider")
    try:
        import onnxruntime as ort
        avail = set(ort.get_available_providers())
        if "CUDAExecutionProvider" in avail:
            prov.append("CUDAExecutionProvider")
        prov = [p for p in prov if p in avail]
    except Exception:
        prov = []
    prov.append("CPUExecutionProvider")
    return prov


def _face(providers=None):
    global _face_app
    if _face_app is not None:
        return _face_app
    from insightface.app import FaceAnalysis
    prov = providers or _default_onnx_providers()
    try:
        app = FaceAnalysis(name="buffalo_l", providers=prov)
    except Exception:
        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    _face_app = app
    return app


def detect_faces(img_bgr):
    """Return [(box_xyxy, embedding_512|None)] for all faces in img. Box is int xyxy."""
    if not FACE_OK:
        return []
    faces = _face().get(img_bgr)
    out = []
    for f in faces:
        x1, y1, x2, y2 = [int(v) for v in f.bbox]
        emb = f.embedding if hasattr(f, "embedding") and f.embedding is not None else None
        out.append(((x1, y1, x2, y2), emb))
    return out


def embed_face(crop_bgr):
    """ArcFace embedding of the largest/best face in a crop, or None."""
    if not FACE_OK:
        return None
    faces = _face().get(crop_bgr)
    if not faces:
        return None
    best = max(faces, key=lambda f: float(getattr(f, "det_score", 0.0)) * (
        (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1])))
    emb = getattr(best, "embedding", None)
    return emb.astype(np.float32) if emb is not None else None


# ---------------- PERSON (OSNet) ----------------
# OSNET_BACKEND selects the ReID variant: x0_25 (default) | x1_0. The x1.0
# checkpoint (hf-mirror kaiyangzhou/osnet) is a strict upgrade on clean person
# crops (PRW benchmark showed x0.25 over-matches badly at 0.55).

def _resolve_osnet_x1():
    if os.environ.get("OSNET_X1_WEIGHTS"):
        return os.path.expanduser(os.environ["OSNET_X1_WEIGHTS"])
    for d in (str(skill_root() / "weights" / "osnet"),
              str(models_dir() / "osnet"),
              os.path.expanduser("~/dev/video-retrieval/weights/osnet")):
        hits = sorted(glob.glob(os.path.join(d, "osnet_x1_0*.pth")))
        if hits:
            return hits[0]
    return str(skill_root() / "weights" / "osnet" / "osnet_x1_0_msmt17.pth")


OSNET_X1_WEIGHTS = _resolve_osnet_x1()
REID_X1_OK = os.path.exists(OSNET_X1_WEIGHTS)
OSNET_BACKEND = os.environ.get("OSNET_BACKEND", "x0_25")   # x0_25 | x1_0


def _osnet():
    global _OSNET
    if _OSNET is not None:
        return _OSNET
    import osnet_loader
    dev = pick_device()
    try:
        model, _ = osnet_loader.build(OSNET_WEIGHTS, device=dev)
    except Exception:
        model, _ = osnet_loader.build(OSNET_WEIGHTS, device="cpu")
        dev = "cpu"
    _OSNET = (model, dev)
    return _OSNET


def _osnet_x1():
    global _OSNET_X1
    if _OSNET_X1 is not None:
        return _OSNET_X1
    # torchreid-native checkpoint (4× OSBlock expansion) -> load via torchreid,
    # NOT osnet_loader (whose minimal OSNet only matches the x0.25 layout).
    import torch
    import torchreid
    model = torchreid.models.build_model(name='osnet_x1_0', num_classes=4101,
                                         loss='softmax', pretrained=False, use_gpu=False)
    sd = torch.load(OSNET_X1_WEIGHTS, map_location='cpu')
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    model.load_state_dict(sd, strict=False)
    model.eval()
    dev = pick_device()
    try:
        model.to(dev)
    except Exception:
        model.to('cpu')
        dev = "cpu"
    _OSNET_X1 = (model, dev)
    return _OSNET_X1


def _embed_osnet(crop_bgr, model, dev):
    import torch
    img = cv2.resize(crop_bgr, (128, 256))  # W,H
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    t = ((t - mean) / std).unsqueeze(0).to(dev)
    with torch.no_grad():
        v = model(t)
    return v.cpu().numpy().ravel().astype(np.float32)


def embed_person(crop_bgr):
    """OSNet 512-d embedding of a person crop (256x128, ImageNet norm).
    Variant per OSNET_BACKEND (x0_25 default; x1_0 when the checkpoint exists)."""
    if OSNET_BACKEND == "x1_0":
        if not REID_X1_OK:
            return embed_clip(crop_bgr) if CLIP_OK else None
        return _embed_osnet(crop_bgr, *_osnet_x1())
    if not REID_OK:
        return embed_clip(crop_bgr) if CLIP_OK else None
    return _embed_osnet(crop_bgr, *_osnet())


# ---------------- CLIP (always-available weak body embedder) ----------------
def _clip():
    global _CLIP
    if _CLIP is not None:
        return _CLIP
    # CLIP on CPU: only a fallback embedder for a handful of crops. Always use
    # clip.load() — it extracts the state_dict from the JIT checkpoint and builds
    # a device-agnostic model, avoiding the hardcoded CUDA device in raw TorchScript.
    import clip
    import PIL.Image as Image
    from torchvision import transforms
    dev = "cpu"
    model, _ = clip.load("ViT-B/32", device="cpu", download_root=_CLIP_ROOT)
    npx = model.visual.input_resolution
    _pre = transforms.Compose([
        transforms.Resize(npx, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(npx), transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130258, 0.27577711)),
    ])
    _CLIP = (model, _pre, dev)
    return _CLIP


def embed_clip(crop_bgr):
    if not CLIP_OK:
        return None
    import torch, PIL.Image as Image
    model, preprocess, dev = _clip()
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    t = preprocess(Image.fromarray(rgb)).unsqueeze(0).to(dev)
    with torch.no_grad():
        v = model.encode_image(t)
    return v.cpu().numpy().ravel().astype(np.float32)


# ---------------- helpers ----------------
def detect_persons(img_bgr, person_model, conf=0.2, imgsz=640, device=None):
    """YOLO person boxes (xyxy) via a loaded person model."""
    r = person_model.predict(img_bgr, conf=conf, imgsz=imgsz, device=pick_device(device),
                             verbose=False)[0]
    out = []
    if r.boxes is not None and len(r.boxes):
        for b in r.boxes.xyxy.cpu().numpy():
            out.append(tuple(int(v) for v in b))
    return out


def _pad_crop(img, box, pad=0.15):
    x1, y1, x2, y2 = box
    H, W = img.shape[:2]
    w, h = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - pad * w)); y1 = max(0, int(y1 - pad * h))
    x2 = min(W, int(x2 + pad * w)); y2 = min(H, int(y2 + pad * h))
    return img[y1:y2, x1:x2], (x1, y1, x2, y2)


def select_ref_subject(ref_bgr, mode="auto", person_model=None, person_device=None):
    """Auto-select the most prominent subject from a reference image.

    Returns dict: {crop, box, kind:'face'|'person', query_vec, all:[{box,kind,area}]}.
    mode 'auto'/'face' prefers a face; falls back to person if none/ mode='person'.
    """
    subjects = []
    # faces first — reuse the embedding detect_faces already computed (re-detecting
    # on a tight crop fails for small faces, so never call embed_face on the crop).
    faces = detect_faces(ref_bgr) if mode in ("auto", "face") and FACE_OK else []
    for (box, emb) in faces:
        x1, y1, x2, y2 = box
        subjects.append({"box": box, "kind": "face", "area": (x2-x1)*(y2-y1), "emb": emb})
    # persons
    if person_model is not None and (mode in ("auto", "person") or not subjects):
        for box in detect_persons(ref_bgr, person_model, device=person_device):
            x1, y1, x2, y2 = box
            subjects.append({"box": box, "kind": "person", "area": (x2-x1)*(y2-y1), "emb": None})
    if not subjects:
        # last resort: treat the whole image as one person crop
        H, W = ref_bgr.shape[:2]
        subjects = [{"box": (0, 0, W, H), "kind": "person", "area": W * H, "emb": None}]
    # pick largest; faces win ties by a bonus (prefer face if reasonably sized)
    best = max(subjects, key=lambda s: s["area"] * (1.0 if s["kind"] == "face" else 1.0))
    crop, _ = _pad_crop(ref_bgr, best["box"])
    qv = best["emb"] if (best["kind"] == "face" and best["emb"] is not None) else (
        embed_face(crop) if best["kind"] == "face" else embed_person(crop))
    return {"crop": crop, "box": best["box"], "kind": best["kind"], "query_vec": qv,
            "all": [{"box": s["box"], "kind": s["kind"], "area": s["area"]} for s in subjects]}


def backend_label(mode, kind):
    """Human label of which embedder will be used for a given kind."""
    if kind == "face":
        return "insightface" if FACE_OK else ("clip" if CLIP_OK else "vlm")
    return "osnet" if REID_OK else ("clip" if CLIP_OK else "vlm")
