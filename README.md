<<<<<<< HEAD
---
base_model: facebook/nllb-200-distilled-600M
library_name: peft
tags:
- base_model:adapter:facebook/nllb-200-distilled-600M
- lora
- transformers
---

# Model Card for Model ID

<!-- Provide a quick summary of what the model is/does. -->



## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->



- **Developed by:** [More Information Needed]
- **Funded by [optional]:** [More Information Needed]
- **Shared by [optional]:** [More Information Needed]
- **Model type:** [More Information Needed]
- **Language(s) (NLP):** [More Information Needed]
- **License:** [More Information Needed]
- **Finetuned from model [optional]:** [More Information Needed]

### Model Sources [optional]

<!-- Provide the basic links for the model. -->

- **Repository:** [More Information Needed]
- **Paper [optional]:** [More Information Needed]
- **Demo [optional]:** [More Information Needed]

## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

### Direct Use

<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->

[More Information Needed]

### Downstream Use [optional]

<!-- This section is for the model use when fine-tuned for a task, or when plugged into a larger ecosystem/app -->

[More Information Needed]

### Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the model will not work well for. -->

[More Information Needed]

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

[More Information Needed]

### Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

Users (both direct and downstream) should be made aware of the risks, biases and limitations of the model. More information needed for further recommendations.

## How to Get Started with the Model

Use the code below to get started with the model.

[More Information Needed]

## Training Details

### Training Data

<!-- This should link to a Dataset Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

[More Information Needed]

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Preprocessing [optional]

[More Information Needed]


#### Training Hyperparameters

- **Training regime:** [More Information Needed] <!--fp32, fp16 mixed precision, bf16 mixed precision, bf16 non-mixed precision, fp16 non-mixed precision, fp8 mixed precision -->

#### Speeds, Sizes, Times [optional]

<!-- This section provides information about throughput, start/end time, checkpoint size if relevant, etc. -->

[More Information Needed]

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data, Factors & Metrics

#### Testing Data

<!-- This should link to a Dataset Card if possible. -->

[More Information Needed]

#### Factors

<!-- These are the things the evaluation is disaggregating by, e.g., subpopulations or domains. -->

[More Information Needed]

#### Metrics

<!-- These are the evaluation metrics being used, ideally with a description of why. -->

[More Information Needed]

### Results

[More Information Needed]

#### Summary



## Model Examination [optional]

<!-- Relevant interpretability work for the model goes here -->

[More Information Needed]

## Environmental Impact

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** [More Information Needed]
- **Hours used:** [More Information Needed]
- **Cloud Provider:** [More Information Needed]
- **Compute Region:** [More Information Needed]
- **Carbon Emitted:** [More Information Needed]

## Technical Specifications [optional]

### Model Architecture and Objective

[More Information Needed]

### Compute Infrastructure

[More Information Needed]

#### Hardware

[More Information Needed]

#### Software

[More Information Needed]

## Citation [optional]

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

[More Information Needed]

**APA:**

[More Information Needed]

## Glossary [optional]

<!-- If relevant, include terms and calculations in this section that can help readers understand the model or model card. -->

[More Information Needed]

## More Information [optional]

[More Information Needed]

## Model Card Authors [optional]

[More Information Needed]

## Model Card Contact

[More Information Needed]
### Framework versions

- PEFT 0.19.1
=======
# Intelligent Traffic & Safety Analytics System

A complete perception-to-insight video analytics pipeline: object detection, multi-object tracking, directional counting, rule-based anomaly detection, movement heatmaps, a trained classifier, and instance segmentation — all running on a standard CPU with no GPU required.

**Final Course Project — Deep Learning & Computer Vision**

---

## Overview

Traffic and pedestrian footage is recorded everywhere, but rarely analyzed. This project takes a video file and automatically produces a structured analytics report: how many objects crossed a line and in which direction, where movement concentrates, and whether any anomalous behaviour occurred.

The system combines **pretrained deep learning models** for perception with an **original analytics layer** built on top of their raw output.

## System Architecture

```
Video file
    ↓
YOLO26n (detection)          ← pretrained CNN, inference only
    ↓
ByteTrack (tracking)         ← classical algorithm, assigns stable IDs
    ↓
Analytics layer              ← custom: counting, heatmap, event logging
    ↓
Anomaly layer                ← custom: wrong-way, loitering, stopped
    ↓
Dashboard + CSV logs + annotated video
```

Two additional components run alongside the main pipeline:

- **Section 6** — a small neural network trained from scratch on geometry + motion features
- **Section 7** — YOLO26n-seg, a second pretrained model demonstrating instance segmentation

## Features

| Feature | Description |
|---|---|
| **Multi-class detection & tracking** | People and vehicles (car, motorcycle, bus, truck) tracked simultaneously with stable IDs |
| **Directional counting** | Per-class, per-direction line-crossing counts, each object counted exactly once |
| **Wrong-way detection** | Flags movement against the established majority direction |
| **Loitering detection** | Flags objects present beyond a time threshold without crossing |
| **Stopped/congestion detection** | Flags objects whose movement drops below a speed threshold |
| **Movement heatmap** | Accumulated density map showing where traffic concentrates |
| **Event logging** | Every crossing and anomaly written to CSV with timestamps |
| **Analytics dashboard** | 4-panel summary figure generated automatically |
| **Trained classifier** | A small PyTorch network trained from scratch, with loss curve and confusion matrix |
| **Instance segmentation** | Pixel-precise masks via a second pretrained model |

## Models Used

| Model | Type | Training status |
|---|---|---|
| **YOLO26n** | Convolutional neural network (detection) | Pretrained on COCO — used for inference only |
| **YOLO26n-seg** | Convolutional neural network (segmentation) | Pretrained on COCO — used for inference only |
| **ByteTrack** | Classical algorithm (Kalman filter + assignment) | Not a neural network — nothing to train |
| **Section 6 classifier** | Feedforward network (MLP) | **Trained from scratch** in this notebook |

> **Note on methodology:** This project uses **no fine-tuning and no transfer learning**. The two YOLO models are loaded with their published weights and never updated. The Section 6 classifier is a separate, randomly-initialized network trained from scratch — it does not reuse YOLO's learned features, only its output class labels as training targets.

## Requirements

```
ultralytics
lap
opencv-python
supervision
yt-dlp
matplotlib
```

PyTorch and NumPy are installed automatically as `ultralytics` dependencies. No GPU is required.

## Installation & Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/traffic-safety-analytics.git
   cd traffic-safety-analytics
   ```

2. Open `traffic_safety_analytics_v6.ipynb` in Jupyter or Google Colab.

3. In **Section 2**, choose a video source:
   ```python
   VIDEO_SOURCE_MODE = "sample"    # auto-downloads a ready-made clip (recommended)
   # or "youtube"  → set YOUTUBE_URL to a specific video
   # or "local"    → set LOCAL_VIDEO_PATH to your own file
   ```

4. Run all cells from top to bottom. Model weights (~5 MB and ~6 MB) download automatically on first run.

**Note:** Live preview is handled automatically — a normal OpenCV window locally (press `q` to stop early), or a self-replacing frame display on Google Colab. The annotated video is saved either way.

## Configuration

All settings are in **Section 3**. Key parameters:

| Parameter | Default | Description |
|---|---|---|
| `CLASSES_TO_COUNT` | `[0, 2, 3, 5, 7]` | COCO class IDs (person, car, motorcycle, bus, truck) |
| `CONF_THRESHOLD` | `0.4` | Minimum detection confidence |
| `LINE_ORIENTATION` | `"horizontal"` | Counting line direction (`"horizontal"` or `"vertical"`) |
| `LINE_POSITION` | `0.5` | Line position as a fraction of frame height/width |
| `LOITER_THRESHOLD_SECONDS` | `8.0` | Time before an object is flagged as loitering |
| `STOPPED_THRESHOLD_PX_PER_SEC` | `5.0` | Speed below which an object is flagged as stopped |
| `WRONG_WAY_MIN_BASELINE_CROSSINGS` | `5` | Crossings needed before wrong-way detection activates |
| `CLASSIFIER_EPOCHS` | `60` | Training epochs for the Section 6 classifier |
| `SEGMENTATION_SAMPLE_FRAMES` | `3` | Frames to run segmentation on |

## Outputs

Running the notebook once produces:

| File | Contents |
|---|---|
| `output_analyzed.mp4` | Full annotated video with boxes, IDs, counting line, and heatmap overlay |
| `crossing_events.csv` | Every line crossing: frame, timestamp, track ID, class, direction |
| `anomaly_events.csv` | Every anomaly: frame, timestamp, track ID, class, type, detail |
| `heatmap_final.png` | Standalone movement density map |
| `dashboard.png` | 4-panel analytics summary |
| `classifier_training.png` | Loss curve and test-set confusion matrix |
| `segmentation_demo.png` | Original vs. segmented frame comparison |

## Notebook Structure

| Section | Purpose |
|---|---|
| 1 | Install and import libraries; detect Colab environment |
| 2 | Obtain a video (sample clip, YouTube, or local file) |
| 3 | Configuration — all tunable parameters |
| 4 | Core engine: detection, tracking, counting, anomaly detection, heatmap |
| 5 | Analytics dashboard generation |
| 6 | Train a small classifier on geometry + motion features |
| 7 | Instance segmentation demonstration |

## Limitations

- Counting and anomaly detection depend on stable tracking IDs; heavy occlusion in crowded scenes can cause ID switches.
- "Stopped" detection uses pixel displacement, not calibrated real-world speed — values are in px/s, not km/h.
- Wrong-way detection requires a baseline of prior crossings, so very short clips may not trigger it.
- The Section 6 classifier's labels come from YOLO26n's own predictions, not independent ground truth — it measures whether geometry and motion alone can recover the pretrained model's classification, not whether it outperforms it.
- Its training rows are frame-level samples from the same tracked objects, so nearby rows are correlated rather than fully independent.
- Segmentation runs on sample frames only, independently of the main pipeline; its results are not fed back into counts or logs.

## Future Work

- Camera calibration for true real-world speed estimates
- Zone-based occupancy tracking in addition to line crossing
- A learned re-identification model for multi-camera tracking
- Class-weighted training for the classifier to address class imbalance

## Credits

- Detection and segmentation models: [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- Sample video clips: [Supervision](https://github.com/roboflow/supervision) by Roboflow
>>>>>>> 27866d832d81527059d0b675c18f7d53454c0e15
