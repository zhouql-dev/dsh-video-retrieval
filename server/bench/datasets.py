#!/usr/bin/env python3
"""Dataset loaders for the four benchmarks (PRW / CCPD / RSTPReid / IJB-A).

All paths default under ``<repo>/dataset/``. Loaders are pure data readers —
no model imports — so they are unit-testable with plain python. Formats were
verified against the actual files (2026-08-13):

  * PRW       : annotations/*.mat `box_new` = (N,5) [id, x, y, w, h] float64;
                frame_test.mat / frame_train.mat hold STRING frame names
                (`img_index_test`, `img_index_train`); query_info.txt =
                `id x y w h frame`; query crops = `query_box/{id}_{frame}.jpg`.
  * CCPD      : filename `{plate}-{angle}-{bbox}-{corners}-{points}-{brightness}-{blur}`,
                bbox = `x1&y1_x2&y2`. ⚠️ THIS balance subset's first field is an
                anonymized numeric ID (no plate-text GT); text GT is UNAVAILABLE
                — only bbox GT. OCR accuracy vs plate text is therefore blocked;
                detection IoU is the primary metric here.
  * RSTPReid  : data_captions.json = list of {id, img_path, captions[2], split
                (train/val/test)}; imgs/{id}_c{cam}_{n}.jpg.
  * IJB-A     : 1:1 = verify_comparisons_N.csv (TEMPLATE_ID pairs) +
                verify_metadata_N.csv (TEMPLATE_ID -> SUBJECT_ID, FILE, face
                box); GT of a pair = same SUBJECT_ID. 1:N = search_gallery_N /
                search_probe_N (TEMPLATE_ID, SUBJECT_ID, FILE, face box).
"""
from __future__ import annotations
import csv
import glob
import json
import os

DATA_ROOT = os.environ.get("BENCH_DATA", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "dataset"))


# --------------------------------------------------------------------------- #
# PRW
# --------------------------------------------------------------------------- #

def load_prw(root: str | None = None, split: str = "test"):
    """Return {gallery: [{id, frame, x, y, w, h}], queries: [{id, frame, x, y, w, h,
    crop_path}]}. Gallery = all GT boxes of the split frames; queries = the
    2,057 official query crops (official protocol uses them all regardless of
    split). Requires scipy (only imported here, lazily)."""
    import numpy as np
    import scipy.io as sio
    root = root or os.path.join(DATA_ROOT, "PRW-v16.04.20")
    split_mat = os.path.join(root, f"frame_{split}.mat")
    frames = [str(n[0]) for n in sio.loadmat(split_mat)[f"img_index_{split}"].ravel()]
    gallery = []
    for fn in frames:
        # annotation files are named `{frame}.jpg.mat` (frames/ holds {frame}.jpg)
        m = sio.loadmat(os.path.join(root, "annotations", fn + ".jpg.mat"))
        # some frames carry no boxes (annotation file without `box_new`)
        for row in m.get("box_new", []):
            pid, x, y, w, h = [float(v) for v in row]
            gallery.append({"id": int(pid), "frame": fn, "x": x, "y": y, "w": w, "h": h})
    queries = []
    with open(os.path.join(root, "query_info.txt")) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            pid, x, y, w, h, frame = int(parts[0]), *map(float, parts[1:5]), parts[5]
            queries.append({"id": pid, "frame": frame, "x": x, "y": y, "w": w, "h": h,
                            "crop_path": os.path.join(root, "query_box",
                                                      f"{pid}_{frame}.jpg")})
    return {"gallery": gallery, "queries": queries, "split": split,
            "n_frames": len(frames)}


# --------------------------------------------------------------------------- #
# CCPD (bbox GT only — see module docstring)
# --------------------------------------------------------------------------- #

def load_ccpd(root: str | None = None, split: str = "valn", limit: int = 0):
    """Return [{path, plate_id (anonymized, NOT text GT), bbox:[x1,y1,x2,y2]}].
    ``plate_id`` is kept only for provenance; do NOT treat it as OCR ground
    truth."""
    root = root or os.path.join(DATA_ROOT, "CCPD")
    out = []
    for path in sorted(glob.glob(os.path.join(root, split, "*.jpg"))):
        name = os.path.basename(path)
        plate_id = name.split("-")[0]
        bbox = name.split("-")[2]                     # x1&y1_x2&y2
        try:
            p1, p2 = bbox.split("_")
            x1, y1 = map(int, p1.split("&"))
            x2, y2 = map(int, p2.split("&"))
        except ValueError:
            continue
        out.append({"path": path, "plate_id": plate_id, "bbox": [x1, y1, x2, y2]})
        if limit and len(out) >= limit:
            break
    return out


# --------------------------------------------------------------------------- #
# RSTPReid
# --------------------------------------------------------------------------- #

def load_rstpreid(root: str | None = None, split: str = "test"):
    """Return [{id, img_path, captions}] for the given split (train/val/test)."""
    root = root or os.path.join(DATA_ROOT, "RSTPReid")
    data = json.load(open(os.path.join(root, "data_captions.json")))
    return [{"id": d["id"], "img_path": os.path.join(root, "imgs", d["img_path"]),
             "captions": d["captions"]} for d in data if d.get("split") == split]


# --------------------------------------------------------------------------- #
# IJB-A
# --------------------------------------------------------------------------- #

def _read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_ijba_metadata(root: str | None = None, split: int = 1) -> dict:
    """verify_metadata -> templates: {template_id: {subject_id, files:[{file,
    bbox}]}} for one split. Face box from FACE_X/Y/WIDTH/HEIGHT (may be empty
    for some files — caller crops by detection then)."""
    root = root or os.path.join(DATA_ROOT, "IJB-A")
    d = os.path.join(root, "protocols", "IJB-A_11_sets", f"split{split}")
    templates = {}
    for row in _read_csv(os.path.join(d, f"verify_metadata_{split}.csv")):
        tid = row["TEMPLATE_ID"]
        t = templates.setdefault(tid, {"subject_id": row["SUBJECT_ID"], "files": []})
        fpath = os.path.join(root, row["FILE"])
        box = None
        try:
            if row.get("FACE_X") not in (None, ""):
                fx, fy = float(row["FACE_X"]), float(row["FACE_Y"])
                fw, fh = float(row["FACE_WIDTH"]), float(row["FACE_HEIGHT"])
                box = [fx, fy, fx + fw, fy + fh]
        except (TypeError, ValueError):
            box = None
        t["files"].append({"file": fpath, "bbox": box})
    return templates


def load_ijba_11(root: str | None = None, split: int = 1) -> dict:
    """1:1 verification for one split: {templates, pairs:[{t1, t2, same}]}
    where same = the two templates share SUBJECT_ID."""
    templates = load_ijba_metadata(root, split)
    d = os.path.join(root or os.path.join(DATA_ROOT, "IJB-A"), "protocols",
                     "IJB-A_11_sets", f"split{split}")
    pairs = []
    # verify_comparisons_N.csv is headerless: two template ids per line.
    with open(os.path.join(d, f"verify_comparisons_{split}.csv")) as f:
        for line in f:
            cols = [c.strip() for c in line.split(",")]
            if len(cols) < 2:
                continue
            t1, t2 = cols[0], cols[1]
            if t1 in templates and t2 in templates:
                pairs.append({"t1": t1, "t2": t2,
                              "same": templates[t1]["subject_id"] == templates[t2]["subject_id"]})
    return {"templates": templates, "pairs": pairs, "split": split}


def load_ijba_1n(root: str | None = None, split: int = 1) -> dict:
    """1:N identification for one split: {templates, gallery:[tid], probe:[tid]}
    GT = same SUBJECT_ID."""
    root = root or os.path.join(DATA_ROOT, "IJB-A")
    d = os.path.join(root, "protocols", "IJB-A_1N_sets", f"split{split}")
    templates = {}
    def add(fname, key):
        for row in _read_csv(os.path.join(d, f"{fname}_{split}.csv")):
            tid = row["TEMPLATE_ID"]
            t = templates.setdefault(tid, {"subject_id": row["SUBJECT_ID"], "files": []})
            fpath = os.path.join(root, row["FILE"])
            box = None
            try:
                if row.get("FACE_X") not in (None, ""):
                    fx, fy = float(row["FACE_X"]), float(row["FACE_Y"])
                    fw, fh = float(row["FACE_WIDTH"]), float(row["FACE_HEIGHT"])
                    box = [fx, fy, fx + fw, fy + fh]
            except (TypeError, ValueError):
                box = None
            t["files"].append({"file": fpath, "bbox": box})
    add("search_gallery", "gallery"); add("search_probe", "probe")
    gallery = [r["TEMPLATE_ID"] for r in _read_csv(os.path.join(d, f"search_gallery_{split}.csv"))]
    probe = [r["TEMPLATE_ID"] for r in _read_csv(os.path.join(d, f"search_probe_{split}.csv"))]
    return {"templates": templates, "gallery": gallery, "probe": probe, "split": split}


if __name__ == "__main__":
    prw = load_prw()
    print("PRW:", prw["split"], "gallery boxes:", len(prw["gallery"]),
          "queries:", len(prw["queries"]))
    print("  sample query:", {k: v for k, v in prw["queries"][0].items() if k != "crop_path"})
    ccpd = load_ccpd(limit=5)
    print("CCPD:", len(ccpd), "sample:", ccpd[0]["bbox"])
    rst = load_rstpreid()
    print("RSTPReid test:", len(rst), "sample captions:", rst[0]["captions"][0][:60])
    ij = load_ijba_11(split=1)
    same = sum(p["same"] for p in ij["pairs"])
    print(f"IJB-A 1:1 split1: {len(ij['templates'])} templates, "
          f"{len(ij['pairs'])} pairs ({same} same, {len(ij['pairs'])-same} diff)")
