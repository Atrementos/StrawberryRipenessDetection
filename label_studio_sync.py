import os
import json
from pathlib import Path
import fiftyone as fo

# --- CONFIGURATION ---
MASTER_DATASET_NAME = "fruit_master_relabel_workflow"
LABEL_FIELD = "ground_truth"
CLASS_NAMES = ["Fullripe", "Semiripe", "Unripe"]

LS_URL = "http://localhost:8080"
LS_API_KEY = "..."

dataset = fo.load_dataset(MASTER_DATASET_NAME)

def push_flagged_to_ls(anno_key="labelstudio_trainval_v1"):
    """Pushes samples marked for relabeling to Label Studio."""
    view = dataset.match(
        fo.ViewField("split").is_in(["train", "val"]) & (fo.ViewField("needs_relabel") == True)
    )
    print(f"Uploading {len(view)} flagged samples...")
    view.annotate(
        anno_key,
        backend="labelstudio",
        label_schema={LABEL_FIELD: {"type": "polygons", "classes": CLASS_NAMES}},
        url=LS_URL,
        api_key=LS_API_KEY,
        launch_editor=True,
    )
    print(f"Upload complete. Annotation key: {anno_key}")

def pull_from_ls_api(anno_key="labelstudio_trainval_v1"):
    """Pulls annotations directly via API."""
    print(f"Pulling annotations for {anno_key}...")
    dataset.load_annotations(anno_key, url=LS_URL, api_key=LS_API_KEY)
    print("Annotations imported successfully.")

def pull_from_ls_json(json_path="./ls_trainval_export.json"):
    """Fallback: Loads annotations from a manual Label Studio JSON export."""
    with open(json_path, "r", encoding="utf-8") as f:
        exported = json.load(f)

    sample_map = {Path(s.filepath).name: s for s in dataset.match(fo.ViewField("split").is_in(["train", "val"]))}
    
    updated = 0
    for task in exported:
        filename = Path(str(task.get("file_upload", ""))).name
        filename = filename.split("-", 1)[1] if "-" in filename else filename
        
        if filename not in sample_map or not task.get("annotations"): continue
        
        latest_ann = sorted(task["annotations"], key=lambda a: a.get("updated_at") or a.get("created_at") or "")[-1]
        
        polylines = []
        for res in latest_ann.get("result", []):
            if res.get("type") == "polygonlabels" and res.get("value", {}).get("points"):
                lbl = res["value"]["polygonlabels"][0]
                pts = [(float(x)/100.0, float(y)/100.0) for x, y in res["value"]["points"] if len(res["value"]["points"]) >= 3]
                if pts and lbl in CLASS_NAMES:
                    polylines.append(fo.Polyline(label=lbl, points=[pts], closed=True, filled=True))
        
        sample = sample_map[filename]
        sample[LABEL_FIELD] = fo.Polylines(polylines=polylines) if polylines else None
        sample["ls_json_imported"] = True
        sample.save()
        updated += 1

    print(f"Updated {updated} samples from JSON.")

# push_flagged_to_ls("trainval_disagreements_v1")
# pull_from_ls_api("trainval_disagreements_v1")
# pull_from_ls_json("./ls_trainval_export.json")