# Strawberry Ripeness Detection Baseline

This repository contains a baseline object detection workflow for **multi-class strawberry ripeness detection** using **Ultralytics YOLO** models. The main goal is to detect strawberries and classify them by ripeness stage:

- **Fullripe**
- **Semiripe**
- **Unripe**

The project is organized around the notebook `strawberrydetectionbaselinev1.ipynb`, which covers dataset preparation, stratified splitting, label distribution checks, hyperparameter experimentation, model training, and qualitative prediction visualization.

YOLO26s weights are available at [Hugging Face](https://huggingface.co/Zenma/StrawberryRipenessDetection-YOLO26s).
## Project Overview

The notebook includes:

- dataset loading from YOLO-format images and labels
- **multilabel stratified** train/validation/test splitting
- YAML generation for Ultralytics training
- bounding box distribution analysis across splits
- hue-jitter (`hsv_h`) hyperparameter search
- final training and evaluation for **YOLOv8s**
- additional training and evaluation for **YOLO26s**
- side-by-side validation visualization of ground truth vs predictions

## Dataset
We used Multi-Class Strawberry Ripeness Detection Dataset from mahyeks which is available at [Kaggle](https://www.kaggle.com/datasets/mahyeks/multi-class-strawberry-ripeness-detection-dataset).

The dataset is split into three subsets:

- **Train:** 398 images
- **Validation:** 82 images
- **Test:** 86 images

### Bounding Box Class Distribution per Split

| Split | Fullripe | Semiripe | Unripe | Total BBoxes |
| ----- | -------: | -------: | -----: | -----------: |
| Train |      351 |      175 |    679 |         1205 |
| Val   |       80 |       39 |    155 |          274 |
| Test  |       89 |       33 |    133 |          255 |

This shows that **Unripe** is the most common class across all splits, while **Semiripe** is the least represented.

## Hue-Jitter Hyperparameter Search

A small experiment was run to test the effect of hue augmentation (`hsv_h`) on validation performance.

| hsv_h | Best Val mAP@50 | Optimal Epoch |
| ----: | --------------: | ------------: |
|  0.00 |        0.744913 |            47 |
|  0.03 |        0.750143 |            65 |
|  0.05 |        0.737152 |            46 |
|  0.10 |        0.738879 |            47 |
|  0.15 |        0.720964 |            35 |
|  0.30 |        0.741903 |            99 |

### Observation

The best validation result in this search was achieved with:

- **`hsv_h = 0.03`**
- **Best Val mAP@50 = 0.750143**
- **Optimal Epoch = 65**

This suggests that a **small amount of hue augmentation** helped performance slightly more than no hue jitter, while stronger hue perturbation did not improve results.

## Final Model Results

### YOLOv8s

Final test metrics for **YOLOv8s**:

- **Mean Precision (mP):** 0.7519
- **Mean Recall (mR):** 0.7586
- **mAP@50:** 0.7821
- **mAP@50-95:** 0.6825

### YOLO26s

Final test metrics for **YOLO26s**:

- **Mean Precision (mP):** 0.8013
- **Mean Recall (mR):** 0.7788
- **mAP@50:** 0.7951
- **mAP@50-95:** 0.7004

### Comparison

In these experiments, **YOLO26s outperformed YOLOv8s** on all reported test metrics, including precision, recall, mAP@50, and mAP@50-95.

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

If you are using a GPU, install a PyTorch build that matches your CUDA version from the official PyTorch installation instructions before running the notebook.

## Dataset Format Assumptions

The notebook expects that dataset directory is in the same folder with the its original structure:

```text
all/
├── images/
│   ├── *.jpg / *.png
└── labels/
    ├── *.txt
```

## Running the Notebook

Launch Jupyter:

```bash
jupyter notebook
```

Then open:

```text
strawberrydetectionbaselinev1.ipynb
```

Run the notebook cells in order.
