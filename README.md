# Triplet Detection Model for Surgical Videos

## Overview
This project focuses on developing a triplet detection model using surgical videos. The model predicts triplet IDs, which include instrument IDs, verb IDs, and target IDs, while also performing spatial detection of instrument tools within video frames. The workflow integrates training object detection models for spatial detection and building a classification model for triplet prediction.

## Workflow

### 1. Spatial Detection of Instrument Tools
- **YOLOv5 Training**: The YOLOv5 model was initially trained on the CholecSeg8k dataset to detect bounding boxes for surgical instruments.
- **Fine-tuning**: The model was fine-tuned on the M2CAI-2016-Tool-Location dataset to improve accuracy, specifically for detecting instrument tips.
  - Trained YOLOv5 weights: `yolov5(2).pt`
  - Fine-tuned YOLOv5 weights: `finetuned.pt`

### 2. Triplet Prediction
A deep learning model was created to predict:
- **Instrument ID**
- **Verb ID**
- **Target ID**
- **Triplet ID**

For training the triplet prediction model, cropped images focusing on the detected instrument tips were used to ensure the model concentrated on relevant regions.

### Inference Workflow
1. YOLOv5 was used to generate bounding box coordinates for instrument tips in the video frames.
2. Cropped image regions containing the instrument tips were passed to the triplet detection model.
3. A pre-trained ResNet50 model was employed for feature extraction, followed by a multi-head classification layer to predict the triplet IDs.

## Model Architecture
Below is the triplet detection model code:

```python
from tensorflow.keras.layers import Input, Resizing, GlobalAveragePooling2D, Concatenate, Dense
from tensorflow.keras.applications import ResNet50
from custom_layers import ScaleBBox, CropToBBox
import tensorflow as tf

# Define the model
def build_model(num_instruments, num_verbs, num_targets, num_triplets):
    image_input = Input(shape=(None, None, 3), name="image_input")
    resize_layer = Resizing(224, 224)(image_input)
    bbox_input = Input(shape=(4,), name="bbox_input")

    scale_bbox_layer = ScaleBBox(target_size=(224, 224))
    bbox_input_scaled = scale_bbox_layer(bbox_input)

    crop_to_bbox_layer = CropToBBox()
    cropped_image = crop_to_bbox_layer([image_input, bbox_input_scaled])

    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    x = base_model(cropped_image)
    x = GlobalAveragePooling2D()(x)

    x = Concatenate()([x, bbox_input_scaled])

    instrument_out = Dense(num_instruments, activation="softmax", name="instrument_id")(x)
    verb_out = Dense(num_verbs, activation="softmax", name="verb_id")(x)
    target_out = Dense(num_targets, activation="softmax", name="target_id")(x)
    triplet_out = Dense(num_triplets, activation="softmax", name="triplet_id")(x)

    return tf.keras.Model(inputs=[image_input, bbox_input], outputs=[instrument_out, verb_out, target_out, triplet_out])

# Compile the model
model = build_model(len(instrument_classes), len(verb_classes), len(target_classes), len(triplet_classes))
```

## Datasets
- **CholecSeg8k**: Used for training YOLOv5 for instrument detection.
- **M2CAI-2016-Tool-Location**: Used for fine-tuning YOLOv5 to improve the detection of instrument tips.
- **CholecT50-Dsg-Subset**: Used for training the triplet prediction model.

## Training Details
- The YOLOv5 fine-tuning process is detailed in `cchole50.ipynb`.
- The triplet prediction model training process is detailed in `cchole50(2).ipynb`.
- `bh-25(1).ipynb` contains incomplete code for modifications to the triplet prediction model, which could not be finalized due to time constraints.

## Inference Code
The inference process is implemented using the following code:

```python
import os
import json
import numpy as np
from tqdm import tqdm
import torch
import tensorflow as tf
from tensorflow.keras import Model
from PIL import Image

# Helper function to preprocess bounding boxes
def process_bbox(bbox_list, image_size=(224, 224)):
    if not bbox_list:
        return [-1, -1, -1, -1]

    bbox = bbox_list[0] if isinstance(bbox_list, list) else bbox_list

    if isinstance(bbox, torch.Tensor):
        bbox = bbox.numpy()

    if len(bbox) == 4:
        x_min, y_min, x_max, y_max = bbox
        image_w, image_h = image_size

        bbox_x = (x_min + x_max) / 2 / image_w
        bbox_y = (y_min + y_max) / 2 / image_h
        bbox_w = (x_max - x_min) / image_w
        bbox_h = (y_max - y_min) / image_h

        return [bbox_x, bbox_y, bbox_w, bbox_h]
    
    return [-1, -1, -1, -1]

# Generate predictions
def generate_predictions_with_decision_tree(model, test_df, instrument_classes, verb_classes, target_classes, triplet_classes, batch_size=32):
    predictions = {}
    for video_id in tqdm(test_df["video_id"].unique(), desc="Processing Videos"):
        video_predictions = {}
        video_frames = test_df[test_df["video_id"] == video_id]

        for start_idx in range(0, len(video_frames), batch_size):
            batch = video_frames.iloc[start_idx:start_idx+batch_size]
            images, bboxes = [], []

            for _, row in batch.iterrows():
                frame_id = os.path.basename(row["frame_path"]).split(".")[0]
                bbox_list = row["bbox"]
                processed_bbox = process_bbox(bbox_list)

                image, bbox, _ = preprocess_image(
                    row["frame_path"], {"bbox": processed_bbox}, len(instrument_classes),
                    len(verb_classes), len(target_classes), len(triplet_classes)
                )
                images.append(image)
                bboxes.append(bbox)

            images_batch = np.array(images)
            bboxes_batch = np.array(bboxes)

            try:
                preds = model.predict([images_batch, bboxes_batch], verbose=1)

                for idx, row in enumerate(batch.iterrows()):
                    frame_id = os.path.basename(row[1]["frame_path"]).split(".")[0]
                    video_predictions[frame_id] = {
                        "recognition": [np.argmax(pred[idx]) for pred in preds[:4]],
                        "detection": bboxes[idx]
                    }
            except Exception as e:
                print(f"Error during model prediction: {e}")

        predictions[video_id] = video_predictions

    return predictions

# Save predictions
output_predictions = generate_predictions_with_decision_tree(
    model, test_df, instrument_classes, verb_classes, target_classes, triplet_classes
)

output_file = f"{MODEL_NAME}_decision_tree.json"
with open(output_file, "w") as f:
    json.dump(output_predictions, f, indent=4, default=lambda x: int(x) if isinstance(x, np.int64) else x)

print(f"Predictions saved to {output_file}")
```

## Results
The model successfully:
- Detected the spatial locations of surgical instruments in video frames.
- Predicted triplet IDs, including instrument, verb, and target IDs.

## Future Improvements
- Experiment with alternative architectures for triplet prediction.
- Optimize the training and inference process for real-time applications.

## Dependencies
- Python 3.8+
- TensorFlow 2.10+
- YOLOv5

## Usage
To train and use the models:
1. Clone the YOLOv5 repository and train it on the CholecSeg8k dataset.
2. Fine-tune YOLOv5 using `cchole50.ipynb` and save the weights as `finetuned.pt`.
3. Train the triplet prediction model using `cchole50(2).ipynb`.
4. Use the inference script provided to generate predictions on test videos.

