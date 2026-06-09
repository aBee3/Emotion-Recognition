## CNN demonstration
Since one of your goals is to **demonstrate how CNNs work**, I would avoid using a "black box" transfer-learning model (ResNet, EfficientNet, etc.) and instead build a CNN from scratch using TensorFlow/Keras or PyTorch.

---

# Recommended Tools

## Option 1 (Recommended): TensorFlow + Keras

Advantages:

* Easy visualization
* CNN layers are explicit
* Easy access to intermediate outputs
* Built-in training loop

```bash
pip install tensorflow matplotlib numpy opencv-python scikit-learn
```

---

## Additional Libraries

### Visualization

```python
import matplotlib.pyplot as plt
```

For:

* filters
* feature maps
* pooling outputs
* confusion matrix

---

### Data Handling

```python
import numpy as np
import pandas as pd
```

---

### Image Processing

```python
import cv2
```

For:

* reading new images
* resizing to 48x48
* grayscale conversion

---

### Metrics

```python
from sklearn.metrics import confusion_matrix
```

---

# Project Roadmap

---

# Phase 1 — Explore Dataset

Show:

## Class Distribution

Count images per emotion.

```python
Angry
Disgust
Fear
Happy
Sad
Surprise
Neutral
```

Visualize:

```python
plt.bar(...)
```

This immediately demonstrates the dataset imbalance.

---

## Show Sample Images

For each class:

```python
7 rows
1 sample per emotion
```

Example:

```text
Angry
😠

Happy
😊

Sad
😢
```

---

# Phase 2 — Explain CNN Components

Use a single image.

---

## 1. Convolution Filter

Take a face image.

Apply:

### Vertical edge filter

```python
[
[-1,0,1],
[-1,0,1],
[-1,0,1]
]
```

Visualize:

Original

↓

Filtered Image

This demonstrates:

* filter/kernel
* convolution

---

## 2. Stride

Show:

Stride = 1

vs

Stride = 2

Visualization:

```text
□□□□□
□□□□□
□□□□□
```

Output size becomes smaller.

Explain:

Output dimension:

\frac{W-F+2P}{S}+1

Where:

* W = input size
* F = filter size
* P = padding
* S = stride

---

## 3. Padding

Show:

### No padding

```text
48x48
↓
46x46
```

### Same padding

```text
48x48
↓
48x48
```

Visualize border zeros.

---

## 4. Pooling

Show:

### Max Pooling

Input:

```text
1 2
4 3
```

Output:

```text
4
```

Visualize pooling on a face image.

---

# Phase 3 — CNN Architecture

A simple architecture:

```python
Input
48x48x1

Conv2D(32)
ReLU

MaxPool

Conv2D(64)
ReLU

MaxPool

Conv2D(128)
ReLU

MaxPool

Flatten

Dense(128)
ReLU

Dense(7)
Softmax
```

Diagram:

```text
Image
 ↓
Conv
 ↓
ReLU
 ↓
Pool
 ↓
Conv
 ↓
Pool
 ↓
Flatten
 ↓
Dense
 ↓
Softmax
```

---

# Phase 4 — Show Feature Maps

This is one of the coolest parts.

Take one image.

After first convolution:

```python
32 feature maps
```

Display:

```python
fig, ax = plt.subplots(4,8)
```

Students can literally see:

* eyes detected
* mouth detected
* edges detected

This answers:

> Where is the network looking?

---

# Phase 5 — Flatten Demonstration

Before flatten:

```text
6 x 6 x 128
```

Visualize:

```python
feature_map.shape
```

After flatten:

```text
4608 neurons
```

Visualization:

```text
Cube
↓
Vector
```

For example:

```python
[0.21,0.54,0.82,...]
```

---

# Phase 6 — Training Algorithm

This is the section most professors want.

---

## Forward Pass

```text
Image
 ↓
Conv
 ↓
ReLU
 ↓
Pool
 ↓
Dense
 ↓
Softmax
```

Output:

```python
[0.01,
 0.05,
 0.12,
 0.72,
 0.04,
 0.03,
 0.03]
```

Prediction:

```python
Happy
```

---

## Loss Function

Use:

### Categorical Cross Entropy

```python
loss =
-Σ y log(ŷ)
```

Visualization:

* prediction wrong → loss large
* prediction correct → loss small

---

## Backpropagation

Explain:

```text
Loss
 ↑
Dense
 ↑
Flatten
 ↑
Pool
 ↑
Conv
```

Gradient flows backwards.

Update:

```python
weight =
weight - lr * gradient
```

---

## Weight Update

Show one filter before training.

Example:

```python
[[0.2,0.1,-0.3],
 ...
]
```

After training:

```python
[[0.4,0.7,-0.5],
 ...
]
```

Students can see filters learning.

---

# Phase 7 — Visualize Attention

This is probably the strongest demo.

Use:

## Grad-CAM

Shows:

```text
Face
 ↓
Heatmap
```

Red regions:

* eyes
* eyebrows
* mouth

This directly answers:

> What parts of the face caused the prediction?

Libraries:

```python
tf-keras-vis
```

or

```python
grad-cam
```

---

# Phase 8 — Evaluation

Generate:

### Accuracy

```python
85%
```

---

### Confusion Matrix

Example:

```text
          Predicted
         A F H S
Actual A
Actual F
Actual H
```

Useful because FER2013 commonly confuses:

* Fear ↔ Surprise
* Sad ↔ Neutral

---

# Phase 9 — Classify New Images

Create:

```python
predict_emotion(image_path)
```

Steps:

```python
Load image
↓
Convert grayscale
↓
Resize 48x48
↓
Normalize
↓
CNN
↓
Softmax
↓
Emotion
```

Output:

```python
Prediction: Happy
Confidence: 93.2%
```

---

# Suggested Folder Structure

```text
EmotionCNN/

dataset/
    train/
    test/

src/
    preprocess.py
    cnn_model.py
    train.py
    evaluate.py
    predict.py
    visualization.py

models/
    emotion_cnn.keras

results/
    filters/
    feature_maps/
    gradcam/
```

---


1. **Convolution filters and feature maps** (what features are extracted)
2. **Flatten + forward/backpropagation** (how learning happens)
3. **Grad-CAM heatmaps** (where the CNN is looking when predicting emotions)

Those three pieces make the CNN much more interpretable than simply reporting accuracy.

# Training the CNN
# What you have now

You built:

```text
Image
 ↓
Padding
 ↓
Convolution
 ↓
Pooling
```

for **one image**.

This was an educational demonstration.

---

# Next: Apply to ALL images

Instead of:

```python
demo_image = X_train[0]
```

you'll do:

```python
for image in X_train:
    ...
```

or let TensorFlow do it automatically.

Conceptually:

```text
Image 1
 ↓
CNN

Image 2
 ↓
CNN

Image 3
 ↓
CNN

...

Image 28709
 ↓
CNN
```

---

# Actual CNN Pipeline

For each image:

```text
48x48 Face
    ↓
Convolution
    ↓
ReLU
    ↓
Pooling
    ↓
Convolution
    ↓
ReLU
    ↓
Pooling
    ↓
Flatten
    ↓
Dense Layer
    ↓
Softmax
```

Notice:

**Softmax happens at the end**, not immediately after convolution.

---

# Step 1: Forward Pass

Take one image.

Suppose:

```text
Happy face
```

True label:

```text
Happy
```

The CNN outputs:

```python
[
 0.01,  # Angry
 0.00,  # Disgust
 0.02,  # Fear
 0.85,  # Happy
 0.05,  # Sad
 0.04,  # Surprise
 0.03   # Neutral
]
```

These are called **logits after softmax**.

The prediction is:

```text
Happy
```

because 0.85 is largest.

---

# Step 2: Softmax

This converts raw scores into probabilities.

Example:

Before:

```python
[2.1, -1.3, 4.7, 0.2]
```

After softmax:

```python
[0.06, 0.00, 0.90, 0.04]
```

Visualization:

```text
Raw Scores
 ↓
Softmax
 ↓
Probabilities
```

For your report:

```text
Angry      1%
Disgust    0%
Fear       2%
Happy     85%
Sad        5%
Surprise   4%
Neutral    3%
```

---

# Step 3: Calculate Loss

Compare:

Prediction:

```python
[0.01,0.00,0.02,0.85,0.05,0.04,0.03]
```

Actual:

```python
[0,0,0,1,0,0,0]
```

Use categorical cross-entropy.

Visualize:

```text
Prediction ≠ Reality
        ↓
     Loss
```

Small loss:

```text
Good prediction
```

Large loss:

```text
Bad prediction
```

---

# Step 4: Backpropagation

Now the network asks:

```text
Why was my prediction wrong?
```

Gradients are computed.

```text
Loss
 ↑
Dense
 ↑
Flatten
 ↑
Pool
 ↑
Conv
```

Error travels backwards.

---

# Step 5: Update Parameters

This is where learning occurs.

Update:

```text
Filters
Biases
Dense Weights
```

using:

```text
new_weight =
old_weight
-
learning_rate × gradient
```

---

# Educational Visualization

Before training:

```text
Filter #1

-0.13  0.24  0.08
 0.11 -0.03  0.42
-0.09  0.17  0.21
```

After epoch 1:

```text
Filter #1

-0.11  0.29  0.15
 0.14 -0.01  0.44
-0.05  0.21  0.25
```

Students can literally see the filter changing.

---

# Step 6: Repeat

One image:

```text
Forward
Loss
Backward
Update
```

is one training step.

After thousands of images:

```text
1 Epoch = All training images seen once
```

For FER2013:

```text
28709 images
=
1 epoch
```

---

# Educational Terminal Output

You could print:

```text
Epoch 1/20

Loss:      1.95
Accuracy:  28%

Epoch 2/20

Loss:      1.73
Accuracy:  37%

Epoch 3/20

Loss:      1.42
Accuracy:  49%

...
```

---

# Step 7: Plot Learning Curves

Very important.

Plot:

```text
Epoch
  ↑
Accuracy
```

and

```text
Epoch
  ↑
Loss
```

Expected:

```text
Accuracy ↑
Loss ↓
```

---

# Step 8: Test Dataset

Once training finishes:

```python
X_test
```

is shown to the CNN.

These images were never used during training.

---

# Step 9: Evaluate Performance

Metrics:

### Accuracy

```text
85%
```

### Confusion Matrix

Example:

```text
            Predicted

          A  F  H  S

Actual A 90  4  2  4
Actual F 10 70 12  8
...
```

This is often the most informative evaluation for FER2013 because some emotions are confused more than others.

---

# Step 10: Visualize Learned Filters

Now revisit the filters.

Show:

```text
Initial Filter
```

vs

```text
Learned Filter
```

Some will become edge detectors.

Some will respond to:

```text
Eyes
Eyebrows
Mouth corners
Smile lines
```

---

# Step 11: Classify New Image

Finally:

```python
predict("my_face.jpg")
```

Pipeline:

```text
New Image
 ↓
Resize 48x48
 ↓
Normalize
 ↓
CNN
 ↓
Softmax
 ↓
Prediction
```

Output:

```text
Happy

Probabilities:

Angry      1%
Disgust    0%
Fear       2%
Happy     91%
Sad        3%
Surprise   2%
Neutral    1%
```

---

So the complete project roadmap is:

```text
PART 1
Educational CNN Demo
────────────────────
Original Image
Padding
Convolution
Pooling
Visualization

PART 2
Real CNN Training
────────────────────
All Images
Forward Pass
Softmax
Loss
Backpropagation
Weight Update
Repeat Epochs

PART 3
Evaluation
────────────────────
Accuracy
Loss Curve
Confusion Matrix
Learned Filters

PART 4
Deployment
────────────────────
Classify New Face
Display Emotion
Display Probabilities
```

That structure will clearly demonstrate both **how a CNN works internally** and **how it learns to recognize emotions**.
