# Triplet Detection Model for Surgical Videos

## Overview
This project aims to develop a triplet detection model using surgical videos. The model predicts triplet IDs, including instrument IDs, verb IDs, and target IDs, along with spatial detection of instrument tools in video frames. The workflow involves training object detection models for spatial detection and building a classification model for triplet prediction.

## Workflow

### 1. Spatial Detection of Instrument Tools
- **YOLOv5 Training**: The YOLOv5 model was initially trained on the CholecSeg8k dataset to detect bounding boxes of instruments in video frames.
- **Fine-tuning**: The model was fine-tuned on the M2CAI-2016-Tool-Location dataset to improve accuracy, specifically for detecting instrument tips.

### 2. Triplet Prediction
A deep learning model was built to predict the following:
- **Instrument ID**
- **Verb ID**
- **Target ID**
- **Triplet ID**

## Model Architecture
Below is the code for the triplet detection model:

```python
from tensorflow.keras.layers import Input, Resizing, GlobalAveragePooling2D, Concatenate, Dense
from tensorflow.keras.applications import ResNet50
from custom_layers import ScaleBBox, CropToBBox  # Custom-defined layers
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
- **CholecSeg8k**: Used for training YOLOv5 to detect instrument bounding boxes.
- **M2CAI-2016-Tool-Location**: Used for fine-tuning YOLOv5 to improve spatial detection of instrument tips.

## Results
The model successfully:
- Detected the spatial locations of surgical instruments in video frames.
- Predicted triplet IDs, including instrument, verb, and target IDs, with high accuracy.

## Future Improvements
- Experiment with other architectures for triplet prediction.
- Optimize the training process for real-time applications.

## Dependencies
- Python 3.8+
- TensorFlow 2.10+
- YOLOv5

## Usage
To train the YOLOv5 model and the triplet detection model, follow these steps:
1. Clone the YOLOv5 repository and train it on the CholecSeg8k dataset.
2. Fine-tune YOLOv5 on the M2CAI-2016-Tool-Location dataset.
3. Use the `build_model` function to create the triplet detection model and train it using the processed dataset.



