import shutil
import yaml
from pathlib import Path
import fiftyone as fo

# --- CONFIGURATION ---
MASTER_DATASET_NAME = "fruit_master_relabel_workflow"
LABEL_FIELD = "ground_truth"
OUT_ROOT = Path("./split_recompiled").resolve()

CLASS_NAMES = ["Fullripe", "Semiripe", "Unripe"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASS_NAMES)}

MODEL_PATH = "yolo26/weights/best.pt"

def export_to_yolo():
    dataset = fo.load_dataset(MASTER_DATASET_NAME)
    
    for split in ["train", "val", "test"]:
        (OUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

    for sample in dataset.iter_samples(progress=True):
        split = sample["split"]
        src_img = Path(sample.filepath).resolve()
        
        dst_img = OUT_ROOT / "images" / split / src_img.name
        dst_txt = OUT_ROOT / "labels" / split / (src_img.stem + ".txt")
        shutil.copy2(src_img, dst_img)

        lines = []
        polylines = sample.get(LABEL_FIELD)
        if polylines and getattr(polylines, "polylines", None):
            for poly in polylines.polylines:
                if poly.label in CLASS_TO_ID and poly.points and poly.points[0]:
                    flat = [max(0.0, min(1.0, float(v))) for pt in poly.points[0] for v in pt]
                    if len(flat) >= 6:
                        lines.append(f"{CLASS_TO_ID[poly.label]} " + " ".join(f"{v:.6f}" for v in flat))

        with open(dst_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n" if lines else "")

    with open(OUT_ROOT / "dataset.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({
            "path": ".", "train": "images/train", "val": "images/val", "test": "images/test",
            "nc": len(CLASS_NAMES), "names": CLASS_NAMES
        }, f, sort_keys=False)

    print(f"Exported rebuilt dataset to: {OUT_ROOT}")

export_to_yolo()