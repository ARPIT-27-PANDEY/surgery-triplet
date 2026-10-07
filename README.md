# Surgical Action Triplet Recognition

A deep learning pipeline for **surgical instrument localization and surgical action triplet recognition** from laparoscopic video frames.

The project combines **YOLOv5-based surgical instrument detection** with a **ResNet50-based multi-head classification model** to recognize surgical actions in the form of structured triplets:

```text
<Instrument, Verb, Target>
```

The overall pipeline first detects and localizes surgical instruments and then uses the localized region together with spatial information to predict the **instrument, verb, target, and complete triplet**.

---

## 📌 Overview

Understanding activities in laparoscopic surgery requires more than simply detecting which instruments are present in a frame.

A surgical action can be represented as a triplet:

```text
<Instrument, Verb, Target>
```

For example:

```text
<Grasper, Grasp, Gallbladder>
```

where:

- **Instrument** represents the surgical tool involved in the action.
- **Verb** represents the action being performed.
- **Target** represents the anatomical structure or object on which the action is performed.

This project develops a two-stage computer vision pipeline:

```text
                    Surgical Video
                          │
                          ▼
                     Video Frames
                          │
                          ▼
                ┌──────────────────┐
                │     YOLOv5       │
                │ Instrument       │
                │ Detection        │
                └────────┬─────────┘
                         │
                    Bounding Box
                         │
                         ▼
               Instrument Region Crop
                         │
                         ▼
                ┌──────────────────┐
                │     ResNet50     │
                │ Feature          │
                │ Extraction       │
                └────────┬─────────┘
                         │
              + Bounding Box Features
                         │
                         ▼
                Feature Fusion Layer
                         │
                         ▼
              Multi-Head Classification
                 ┌───────┼────────┐
                 │       │        │
                 ▼       ▼        ▼
            Instrument  Verb    Target
                 │       │        │
                 └───────┴────────┘
                         │
                         ▼
                    Triplet ID
```

The goal is to combine **spatial localization** and **fine-grained surgical action recognition** into a unified workflow.

---

## 🎯 Objectives

The major objectives of the project are:

- Detect surgical instruments in laparoscopic images.
- Improve localization of surgical instrument regions and instrument tips.
- Use localized instrument regions for downstream action recognition.
- Learn visual representations using a pretrained ResNet50 backbone.
- Incorporate bounding-box information together with visual features.
- Predict the individual components of a surgical action.
- Perform joint prediction of **instrument, verb, target, and triplet IDs**.
- Process surgical videos frame-by-frame and generate structured predictions.

---

# 🧠 Methodology

The project follows a two-stage architecture.

## Stage 1 — Surgical Instrument Detection

The first stage identifies the location of surgical instruments in each image.

A **YOLOv5** object detection model is used for this purpose.

The detector produces bounding boxes around the surgical instrument regions:

```text
Input Frame
    │
    ▼
YOLOv5
    │
    ▼
Instrument Bounding Box
```

The detected bounding box is then used to identify the region of interest for the second stage.

### Detector Training

The YOLOv5 model is trained using the **CholecSeg8k** dataset.

The trained model is then further fine-tuned using the **M2CAI-2016-Tool-Location** dataset to improve surgical instrument localization.

The repository contains trained model weights:

```text
yolov5s (2).pt
```

and:

```text
finetuned.pt
```

where the latter corresponds to the fine-tuned detector.

---

# 🔬 Stage 2 — Surgical Triplet Recognition

After obtaining the instrument bounding box, the detected region is passed to a classification network.

A **ResNet50** backbone is used for visual feature extraction.

The pipeline is:

```text
Detected Instrument Region
           │
           ▼
      ResNet50
           │
           ▼
   Visual Feature Vector
           │
           ├──────────────┐
           │              │
           ▼              ▼
 Visual Features     Bounding Box
           │              │
           └──────┬───────┘
                  │
                  ▼
             Feature Fusion
                  │
                  ▼
        Multi-Head Classifier
```

The classifier predicts four related outputs:

```text
1. Instrument ID
2. Verb ID
3. Target ID
4. Triplet ID
```

---

# 🏗️ Model Architecture

## ResNet50 Backbone

A pretrained **ResNet50** network is used to extract high-level visual representations from the localized surgical instrument region.

The classification pipeline can be represented as:

```text
Input Image
     │
     ▼
Crop using Detection Box
     │
     ▼
Resize / Preprocessing
     │
     ▼
ResNet50
     │
     ▼
Global Average Pooling
     │
     ▼
Visual Feature Representation
```

---

## 📦 Bounding Box Features

In addition to visual information extracted by ResNet50, the model uses the spatial information of the detected instrument.

The bounding box is represented as:

```text
[x_center, y_center, width, height]
```

and normalized with respect to the image dimensions.

The spatial features are then combined with the visual features:

```text
ResNet50 Features
        +
Bounding Box Features
        │
        ▼
Feature Fusion
```

This provides the classifier with both:

- **What the region looks like**
- **Where the region is located**

---

# 🎯 Multi-Head Classification

The fused representation is passed to multiple classification heads.

```text
                    Fused Feature
                          │
          ┌───────────────┼────────────────┐
          │               │                │
          ▼               ▼                ▼
   Instrument Head    Verb Head       Target Head
          │               │                │
          ▼               ▼                ▼
   Instrument ID       Verb ID          Target ID
                          │
                          ▼
                    Triplet Head
                          │
                          ▼
                      Triplet ID
```

Each head performs a classification task using a softmax output layer.

Conceptually:

```python
instrument_output = Dense(
    num_instruments,
    activation="softmax"
)(features)

verb_output = Dense(
    num_verbs,
    activation="softmax"
)(features)

target_output = Dense(
    num_targets,
    activation="softmax"
)(features)

triplet_output = Dense(
    num_triplets,
    activation="softmax"
)(features)
```

This allows the model to learn both the individual components and the complete surgical triplet.

---

# 📊 Multi-Task Learning

The model performs several related classification tasks simultaneously.

The overall training objective can be represented conceptually as:

```text
Total Loss
    =
Instrument Classification Loss
    +
Verb Classification Loss
    +
Target Classification Loss
    +
Triplet Classification Loss
```

Learning these related tasks jointly encourages the model to learn representations useful for fine-grained surgical action recognition.

---

# 📚 Datasets

Different datasets are used for different stages of the project.

## 1. CholecSeg8k

**Purpose:**

- Initial training of the YOLOv5 surgical instrument detector.
- Learning spatial representations of surgical instruments.

The dataset provides laparoscopic images together with segmentation/instrument information useful for training the detection pipeline.

---

## 2. M2CAI-2016-Tool-Location

**Purpose:**

- Fine-tuning the YOLOv5 detector.
- Improving instrument localization.
- Improving localization around relevant instrument regions/tips.

The fine-tuned detector is saved in the repository as:

```text
finetuned.pt
```

---

## 3. CholecT50

**Purpose:**

- Training and evaluating surgical action triplet recognition.
- Learning the relationship between instruments, verbs, and targets.

The dataset provides annotations associated with surgical actions, including information such as:

```text
Video ID
Frame ID
Triplet ID
Instrument ID
Verb ID
Target ID
Phase ID
Bounding Box
```

The repository notebooks use these annotations to construct training and testing data.

---

# 🧪 Data Preparation

The CholecT50 annotation files are parsed and transformed into structured tabular data.

For each annotated frame, information such as the following is extracted:

```text
Frame
Triplet
Instrument
Verb
Target
Phase
Bounding Box
```

These annotations can then be converted into Pandas DataFrames and exported as CSV files for model training.

Example files include:

```text
train_data.csv
test_data.csv
```

---

# 🎬 Train/Test Split

The current notebook implementation uses a video-wise split for CholecT50.

## Training Videos

```text
VID01
VID02
VID04
VID05
VID06
VID08
VID10
VID12
VID13
VID14
```

## Testing Videos

```text
VID92
VID96
VID103
VID110
VID111
```

The split is performed at the video level so that training and testing examples originate from different surgical videos.

---

# 📍 Bounding Box Processing

The object detector produces bounding boxes in the form:

```text
[x_min, y_min, x_max, y_max]
```

These coordinates are converted into normalized center-width-height representation:

```text
x_center = (x_min + x_max) / (2 × image_width)

y_center = (y_min + y_max) / (2 × image_height)

width = (x_max - x_min) / image_width

height = (y_max - y_min) / image_height
```

Therefore, the bounding box supplied to the classifier is:

```text
[x_center, y_center, width, height]
```

If a valid detection is unavailable, a fallback representation is used:

```text
[-1, -1, -1, -1]
```

---

# 🔄 End-to-End Pipeline

The complete inference process is:

```text
                  Input Surgical Video
                           │
                           ▼
                    Extract Frames
                           │
                           ▼
                  YOLOv5 Detection
                           │
                           ▼
                Instrument Bounding Box
                           │
                           ▼
                  Region of Interest
                           │
                           ▼
                    ResNet50 Backbone
                           │
                           ▼
                  Visual Feature Vector
                           │
               ┌───────────┴───────────┐
               │                       │
               ▼                       ▼
        Visual Features         Bounding Box
               │                       │
               └───────────┬───────────┘
                           │
                           ▼
                     Feature Fusion
                           │
                           ▼
                  Multi-Head Network
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
     Instrument           Verb            Target
          │                │                │
          └────────────────┴────────────────┘
                           │
                           ▼
                      Triplet ID
                           │
                           ▼
                   Frame-wise Output
                           │
                           ▼
                         JSON
```

---

# 🚀 Inference

The inference pipeline operates on surgical video frames.

For every processed frame:

1. The image is passed through the YOLOv5 detector.
2. Instrument bounding boxes are obtained.
3. The relevant region is cropped.
4. Bounding-box coordinates are normalized.
5. The cropped image is passed through ResNet50.
6. Visual and spatial features are combined.
7. The multi-head classifier predicts the surgical action.
8. Predictions are stored for the corresponding frame.

---

# 📝 Prediction Format

The inference pipeline stores frame-level results in JSON format.

The prediction contains information corresponding to recognition and detection.

A representative output structure is:

```json
{
    "recognition": [
        "instrument_prediction",
        "verb_prediction",
        "target_prediction",
        "triplet_prediction"
    ],
    "detection": [
        "x",
        "y",
        "width",
        "height"
    ]
}
```

The generated prediction files follow the general naming convention:

```text
<MODEL_NAME>_decision_tree.json
```

---

# 📈 Results

The project demonstrates a complete pipeline for:

### Instrument Detection

- Localization of surgical instruments in laparoscopic frames.
- Generation of bounding boxes for downstream recognition.

### Triplet Recognition

Prediction of:

```text
Instrument ID
Verb ID
Target ID
Triplet ID
```

This allows the system to move beyond simple object detection and recognize the underlying surgical action.

---

# 🔍 Example

Consider a frame where a grasper interacts with a tissue structure.

The system performs:

```text
Input Frame
     │
     ▼
YOLOv5
     │
     ▼
Grasper Bounding Box
     │
     ▼
Crop Instrument Region
     │
     ▼
ResNet50 Feature Extraction
     │
     ▼
Feature + Spatial Information
     │
     ▼
Multi-Head Classification
     │
     ├── Instrument → Grasper
     ├── Verb       → Grasp
     ├── Target     → Tissue
     └── Triplet    → <Grasper, Grasp, Tissue>
```

This structured representation provides a more detailed interpretation of the surgical activity.

---

# 📁 Repository Structure

```text
surgery-triplet/
│
├── README.md
│
├── cchole50.ipynb
│   └── Dataset preparation and YOLOv5
│       training/fine-tuning experiments
│
├── trained-yolov5.ipynb
│   └── YOLOv5 detector training workflow
│
├── cchole50 (2).ipynb
│   └── Triplet classification and prediction experiments
│
├── bh-25 (1).ipynb
│   └── Experimental modifications to the
│       triplet prediction architecture
│
├── yolov5s (2).pt
│   └── Trained YOLOv5 model weights
│
└── finetuned.pt
    └── Fine-tuned YOLOv5 model weights
```

> **Note:** `bh-25 (1).ipynb` contains experimental modifications to the triplet model. Parts of this notebook are incomplete and are retained as part of the research experimentation process.

---

# 🛠️ Technologies Used

The project combines multiple deep learning and computer vision frameworks.

### Programming

- Python

### Deep Learning

- PyTorch
- TensorFlow
- Keras

### Computer Vision

- YOLOv5
- ResNet50
- Torchvision
- Pillow
- Albumentations

### Data Processing

- NumPy
- Pandas

### Development

- Jupyter Notebook
- PyTorch Lightning

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/ARPIT-27-PANDEY/surgery-triplet.git
cd surgery-triplet
```

Install the major dependencies:

```bash
pip install torch torchvision
pip install tensorflow
pip install pandas numpy pillow
pip install tqdm
pip install albumentations
pip install pytorch-lightning
```

The exact package versions may need to be adapted according to the CUDA/Python environment used for training.

---

# 💾 Dataset Setup

The datasets are not included directly in this repository.

Download and prepare:

```text
CholecSeg8k
M2CAI-2016-Tool-Location
CholecT50
```

After downloading the datasets, update the paths used in the notebooks.

For example, the current notebooks contain Kaggle-style paths similar to:

```text
/kaggle/input/cholec-train-data/CholecT50/
```

These paths should be modified when running locally.

---

# ▶️ Running the Project

## Step 1 — Train YOLOv5

Open:

```text
trained-yolov5.ipynb
```

and execute the detector training pipeline.

The resulting model weights are saved as:

```text
yolov5s (2).pt
```

---

## Step 2 — Fine-Tune the Detector

Open:

```text
cchole50.ipynb
```

and run the fine-tuning pipeline using the M2CAI-2016-Tool-Location dataset.

The fine-tuned weights are saved as:

```text
finetuned.pt
```

---

## Step 3 — Prepare Triplet Data

Use:

```text
cchole50 (2).ipynb
```

to process the CholecT50 annotations and construct the training/testing dataset for the triplet classifier.

---

## Step 4 — Train the Triplet Classifier

The same notebook contains the ResNet50-based multi-head classification workflow.

The model learns:

```text
Instrument
Verb
Target
Triplet
```

from the localized surgical regions and corresponding spatial information.

---

## Step 5 — Run Inference

The inference pipeline can then be used to generate predictions for the test videos.

The output is stored in JSON format for frame-level analysis.

---

# 🧩 Why a Two-Stage Pipeline?

Surgical action recognition is highly dependent on spatial context.

Simply classifying the entire laparoscopic frame may introduce irrelevant background information and make fine-grained action recognition more difficult.

The two-stage approach addresses this by separating:

```text
Localization
     +
Recognition
```

The detector answers:

> **Where is the relevant surgical instrument?**

The classifier then answers:

> **What action involving that instrument is taking place?**

This separation allows the recognition model to focus more directly on the relevant region.

---

# 🔬 Research Perspective

The project investigates surgical video understanding through the combination of:

```text
Object Detection
      +
Region-Based Feature Extraction
      +
Spatial Information
      +
Multi-Task Learning
      ↓
Surgical Action Recognition
```

Instead of treating surgical action recognition as a single classification problem, the action is decomposed into meaningful semantic components:

```text
Instrument + Verb + Target
```

This provides a more interpretable representation of surgical activities.

---

# 📌 Key Components

| Component | Role |
|---|---|
| **YOLOv5** | Surgical instrument detection |
| **CholecSeg8k** | Initial detector training |
| **M2CAI-2016-Tool-Location** | Detector fine-tuning |
| **ResNet50** | Visual feature extraction |
| **Bounding Box Features** | Spatial information |
| **CholecT50** | Surgical action triplet annotations |
| **Multi-Head Classifier** | Instrument, verb, target, and triplet prediction |
| **JSON Output** | Frame-level inference results |

---

# 📚 Related Concepts

## Surgical Instrument Detection

Object detection provides the spatial location of surgical instruments and enables the subsequent recognition stage to focus on relevant regions.

## Surgical Action Recognition

Surgical actions can be represented using:

```text
<Instrument, Verb, Target>
```

This provides semantic information about both the instrument and the action being performed on a particular target.

## Multi-Task Learning

Predicting instrument, verb, target, and triplet simultaneously allows the network to learn multiple related tasks from a shared feature representation.

---

# 🔮 Future Improvements

Several directions can further improve the current implementation:

- Incorporating temporal information from consecutive video frames.
- Improving instrument-tip localization.
- Handling multiple simultaneous instrument detections.
- Improving association between detected instruments and triplet annotations.
- Exploring stronger visual backbones.
- Jointly training the detection and recognition stages.
- Optimizing the complete pipeline for real-time inference.
- Exploring transformer-based architectures for surgical video understanding.
- Evaluating the model using standardized surgical action recognition metrics.

---

# 📖 References

### CholecT50

CholecT50 is a benchmark dataset for surgical action triplet recognition in laparoscopic cholecystectomy videos.

The core representation is:

```text
<Instrument, Verb, Target>
```

---

### YOLOv5

YOLOv5 is an object detection framework used in the first stage of this project.

Repository:

https://github.com/ultralytics/yolov5

---

### ResNet

He et al.,

**Deep Residual Learning for Image Recognition**

The ResNet50 architecture is used as the visual feature extraction backbone for the triplet recognition stage.

---

# 👨‍💻 Author

**Arpit Kumar Pandey**

Indian Institute of Technology Roorkee

GitHub:

https://github.com/ARPIT-27-PANDEY

---

# 🔗 Repository

https://github.com/ARPIT-27-PANDEY/surgery-triplet

---

## ⭐ Project Summary

```text
YOLOv5
   │
   ▼
Surgical Instrument Localization
   │
   ▼
Instrument Region
   │
   ▼
ResNet50 Feature Extraction
   │
   +
Bounding Box Features
   │
   ▼
Multi-Head Classification
   │
   ├── Instrument
   ├── Verb
   ├── Target
   └── Triplet
   │
   ▼
Surgical Action Recognition
```

The project demonstrates an end-to-end approach for **surgical instrument detection and fine-grained surgical action triplet recognition from laparoscopic video**.
