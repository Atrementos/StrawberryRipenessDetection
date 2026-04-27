import os
import yaml
import numpy as np
from pathlib import Path
import fiftyone as fo
import fiftyone.core.fields as fof
from ultralytics import YOLO

# --- CONFIGURATION ---
DATASET_ROOT = Path("./split_original").resolve()
MASTER_DATASET_NAME = "fruit_master_relabel_workflow"
TEST_RELABELED_DATASET_NAME = "fruit_test_for_relabel"
LABEL_FIELD = "ground_truth"
PRED_FIELD = "model_pred_boxes"
MODEL_PATH = "yolo26/weights/best.pt"

MATCH_IOU = 0.5
LOW_IOU_MIN, LOW_IOU_MAX = 0.1, 0.4
CONF_UNCERTAIN = 0.45
PRED_CONF = 0.25


def parse_yolo_polygon_txt(txt_path, class_names):
    if not txt_path.exists(): return None
    polylines = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7: continue
            try:
                class_id = int(float(parts[0]))
                coords = list(map(float, parts[1:]))
            except Exception: continue

            if len(coords) % 2 != 0 or not (0 <= class_id < len(class_names)): continue

            points = [(max(0.0, min(1.0, coords[i])), max(0.0, min(1.0, coords[i + 1]))) 
                      for i in range(0, len(coords), 2)]
            
            if len(points) >= 3:
                polylines.append(fo.Polyline(label=class_names[class_id], points=[points], closed=True, filled=True))
                
    return fo.Polylines(polylines=polylines) if polylines else None

def box_iou_xyxy(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])) + (max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])) - inter
    return inter / union if union > 0 else 0.0

def process_sample_predictions(sample, result, class_names):
    gt_objs, pred_objs = [], []
    
    # Extract GT boxes
    if LABEL_FIELD in sample and getattr(sample[LABEL_FIELD], "polylines", None):
        for idx, poly in enumerate(sample[LABEL_FIELD].polylines):
            pts = np.asarray(poly.points[0], dtype=float)
            if len(pts) >= 3:
                gt_objs.append({"id": f"gt_{idx}", "label": poly.label, 
                                "box": [float(np.min(pts[:, 0])), float(np.min(pts[:, 1])), 
                                        float(np.max(pts[:, 0])), float(np.max(pts[:, 1]))]})
    
    # Extract Pred boxes
    if result.boxes and len(result.boxes) > 0:
        for i, box in enumerate(result.boxes.xyxyn.cpu().numpy()):
            cls_id = int(result.boxes.cls[i])
            if 0 <= cls_id < len(class_names):
                pred_objs.append({"id": f"pred_{i}", "label": class_names[cls_id], 
                                  "conf": float(result.boxes.conf[i]), "box": box.tolist()})

    # Add detections to FiftyOne
    dets = [fo.Detection(label=p["label"], bounding_box=[p["box"][0], p["box"][1], p["box"][2]-p["box"][0], p["box"][3]-p["box"][1]], confidence=p["conf"]) for p in pred_objs]
    sample[PRED_FIELD] = fo.Detections(detections=dets) if dets else None

    # Greedy Match
    pairs = sorted([(box_iou_xyxy(g["box"], p["box"]), gi, pi) for gi, g in enumerate(gt_objs) for pi, p in enumerate(pred_objs)], reverse=True, key=lambda x: x[0])
    used_g, used_p, matches = set(), set(), []
    for iou, gi, pi in pairs:
        if gi not in used_g and pi not in used_p and iou > 0:
            used_g.add(gi); used_p.add(pi); matches.append((gi, pi, iou))

    reasons = []
    for gi, pi, iou in matches:
        if gt_objs[gi]["label"] != pred_objs[pi]["label"] and iou > MATCH_IOU: reasons.append("class_mismatch")
        elif gt_objs[gi]["label"] == pred_objs[pi]["label"] and LOW_IOU_MIN <= iou <= LOW_IOU_MAX: reasons.append("low_iou")
        elif pred_objs[pi]["conf"] < CONF_UNCERTAIN: reasons.append("low_confidence")

    reasons.extend(["missing_prediction"] * (len(gt_objs) - len(used_g)))
    reasons.extend(["false_positive"] * (len(pred_objs) - len(used_p)))

    sample["needs_relabel"] = len(reasons) > 0
    sample["relabel_reasons"] = sorted(set(reasons))

# --- MAIN EXECUTION ---
with open(DATASET_ROOT / "dataset.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
    class_names = [cfg["names"][k] for k in sorted(cfg["names"].keys(), key=int)] if isinstance(cfg["names"], dict) else list(cfg["names"])

if MASTER_DATASET_NAME in fo.list_datasets(): fo.delete_dataset(MASTER_DATASET_NAME)
dataset = fo.Dataset(MASTER_DATASET_NAME)
dataset.persistent = True

# Load Images & Polygons
all_samples = []
for split in ("train", "val", "test"):
    img_dir = (DATASET_ROOT / cfg[split]).resolve()
    lbl_dir = (DATASET_ROOT / "labels" / split).resolve()
    for img_path in img_dir.rglob("*"):
        if img_path.suffix.lower() in {".jpg", ".png", ".jpeg"}:
            sample = fo.Sample(filepath=str(img_path), split=split)
            labels = parse_yolo_polygon_txt((lbl_dir / img_path.relative_to(img_dir)).with_suffix(".txt"), class_names)
            if labels: sample[LABEL_FIELD] = labels
            all_samples.append(sample)
dataset.add_samples(all_samples)

# Merge previously relabeled test set
if TEST_RELABELED_DATASET_NAME in fo.list_datasets():
    test_map = {Path(s.filepath).resolve().as_posix(): s for s in fo.load_dataset(TEST_RELABELED_DATASET_NAME)}
    for s in dataset.match(fo.ViewField("split") == "test"):
        if (key := Path(s.filepath).resolve().as_posix()) in test_map:
            s[LABEL_FIELD] = test_map[key].get(LABEL_FIELD, None)
            s["was_test_relabeled"] = True
            s.save()

# Flag Train/Val Disagreements
if dataset.get_field_schema().get(PRED_FIELD) is None:
    dataset.add_sample_field(PRED_FIELD, fof.EmbeddedDocumentField, embedded_doc_type=fo.Detections)

model = YOLO(MODEL_PATH)
view = dataset.match(fo.ViewField("split").is_in(["train", "val"]))
for sample in view.iter_samples(progress=True, autosave=True):
    process_sample_predictions(sample, model.predict(sample.filepath, conf=PRED_CONF, verbose=False)[0], class_names)

print(f"Flagged samples: {len(dataset.match(fo.ViewField('needs_relabel') == True))}")