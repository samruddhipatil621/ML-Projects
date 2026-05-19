# ✍️ Task 3: Handwritten Character Recognition

Teach a computer to read handwritten digits — just like how your phone reads handwriting.

---

## 🎯 Objective
Build a model that looks at a 28×28 pixel image of a handwritten digit
and correctly identifies which digit (0–9) it is.

---

## 🧠 Concepts Covered (Explained Simply)

| Concept | What it means |
|---|---|
| **CNN** | Convolutional Neural Network — scans images in small patches to find patterns |
| **ANN** | Artificial Neural Network — simpler baseline; treats image as flat list of pixels |
| **Conv2D** | A filter that slides over the image and detects edges, curves, corners |
| **MaxPooling** | Shrinks the feature map — keeps only the most important information |
| **Dropout** | Randomly turns off neurons during training — prevents memorization |
| **BatchNorm** | Normalizes layer outputs — makes training faster and more stable |
| **ReLU** | Activation function — introduces non-linearity into the model |
| **Softmax** | Converts raw scores into probabilities (all sum to 1.0) |
| **One-hot encoding** | "3" → [0,0,0,1,0,0,0,0,0,0] — the output format for classification |
| **Data Augmentation** | Applies random small transforms to training images — more variety |
| **Early Stopping** | Stops training when model stops improving — saves time |
| **Overfitting** | Model memorizes training data but fails on new images — we prevent this |

---

## 📁 Project Structure
```
Task3_Handwritten_Recognition/
├── handwritten_recognition.py  ← Main code (run this)
├── requirements.txt            ← Python packages needed
├── README.md                   ← This file
└── outputs/                    ← Auto-created after running
    ├── sample_images.png
    ├── pixel_analysis.png
    ├── evaluation_dashboard.png
    ├── cnn_filters.png
    ├── demo_predictions.png
    └── handwriting_cnn_model.keras
```

---

## 📊 Dataset — MNIST

| Property | Value |
|---|---|
| **Name** | MNIST (Modified National Institute of Standards & Technology) |
| **Images** | 70,000 handwritten digit images |
| **Training** | 60,000 images |
| **Testing** | 10,000 images |
| **Image size** | 28 × 28 pixels (784 total pixels per image) |
| **Colour** | Grayscale (0 = black, 255 = white) |
| **Classes** | 10 (digits 0–9) |
| **Download** | Automatic via Keras — no Kaggle account needed |
| **Size** | ~11 MB |

### What one image looks like
```
Each image is a 28×28 grid of numbers, like this (simplified):

  0   0   0   0   0   0   0   0   0   0   0   0
  0   0   0   0   0   0   0   0   0   0   0   0
  0   0   0  50 200 250 210  80   0   0   0   0   ← pixels of a "7"
  0   0   0   0   0  30 180 240  90   0   0   0
  0   0   0   0   0   0  10 200 180   0   0   0
  ...
```

---

## 🏗️ Model Architecture

### ANN (Baseline)
```
Input (784) → Dense(256, ReLU) → Dense(128, ReLU) → Dense(64, ReLU) → Output(10, Softmax)
```
- Treats the 28×28 image as a flat list of 784 numbers
- Does NOT understand spatial structure
- Expected accuracy: ~97–98%

### CNN (Main Model)
```
Input (28×28×1)
    ↓
[Conv2D(32) → BatchNorm → Conv2D(32) → MaxPool → Dropout]    ← detects edges/lines
    ↓
[Conv2D(64) → BatchNorm → Conv2D(64) → MaxPool → Dropout]    ← detects curves/shapes
    ↓
[Conv2D(128) → BatchNorm → MaxPool → Dropout]                 ← detects digit parts
    ↓
Flatten → Dense(512) → Dense(256) → Output(10, Softmax)
```
- Scans the image in 3×3 patches — understands 2D structure
- Expected accuracy: **99%+**

---

## ⚙️ Setup & Run

### Step 1 — Install packages
```bash
pip install -r requirements.txt
```

### Step 2 — Run the code
```bash
python handwritten_recognition.py
```

The MNIST dataset downloads automatically (~11 MB) on the first run. Training takes **5–15 minutes** depending on your computer.

> **No GPU?** It still works on CPU — just slower. Reduce `EPOCHS = 15` in the code to make it faster.

### Step 3 — View outputs
Open the generated `.png` files in the same folder to see all charts.

---

## 📈 Expected Results

| Model | Expected Accuracy |
|---|---|
| ANN (Simple Neural Network) | ~97–98% |
| CNN (Convolutional Neural Net) | ~99–99.5% |

MNIST is so well-studied that 99%+ is the standard benchmark for CNNs.
On a dataset of 10,000 test images — 99% means only 100 mistakes!

---

## 🎛️ Configuration Options (top of file)

```python
DATASET    = "mnist"   # "mnist" for digits, "emnist" for A-Z letters
EPOCHS     = 30        # How many training rounds (EarlyStopping stops early)
BATCH_SIZE = 128       # Images per training step
```

### Switch to EMNIST (Letters A–Z)
1. Install extra package: `pip install extra-keras-datasets`
2. Change `DATASET = "emnist"` in the code
3. Run again — now trains on 26 letter classes!

---

## ✍️ Predict Your Own Handwriting

1. Open **MS Paint** (Windows) or **Preview** (Mac)
2. Create new image → resize to **28×28 pixels**
3. Fill background with **black** (#000000)
4. Draw a white digit in the center
5. Save as `my_digit.png` in the same folder
6. Uncomment the last line in the code:
```python
predict_custom_image("my_digit.png", cnn_model, class_names)
```
7. Run — the model tells you what it sees!

---

## 📊 Output Files Explained

| File | What it shows |
|---|---|
| `sample_images.png` | Two sample images per digit class |
| `pixel_analysis.png` | Pixel distribution + average digit appearance |
| `evaluation_dashboard.png` | Full 6-panel result: accuracy curves, loss, confusion matrix, correct/wrong examples |
| `cnn_filters.png` | The 32 patterns the CNN learned to detect (edges, textures) |
| `demo_predictions.png` | One prediction per class with confidence % |
| `handwriting_cnn_model.keras` | The trained model — load and reuse anytime |

---

## 🔍 How to Load the Saved Model Later

```python
import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model("handwriting_cnn_model.keras")

# Prepare your image (must be 28×28 grayscale, normalized)
img = your_image.reshape(1, 28, 28, 1).astype("float32") / 255.0

# Predict
probs = model.predict(img)[0]
digit = np.argmax(probs)
print(f"Predicted digit: {digit}  Confidence: {probs[digit]*100:.1f}%")
```

---

## 📚 Resources to Learn More

- [MNIST on Wikipedia](https://en.wikipedia.org/wiki/MNIST_database)
- [CNN Explained Visually](https://poloclub.github.io/cnn-explainer/) — highly recommended!
- [Keras Documentation](https://keras.io/)
- [3Blue1Brown: Neural Networks series](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)

---

## ⚠️ Common Errors

| Error | Fix |
|---|---|
| `ModuleNotFoundError: tensorflow` | `pip install tensorflow` |
| Training too slow | Set `EPOCHS = 10` in config |
| `ModuleNotFoundError: PIL` | `pip install Pillow` |
| Custom image not recognized | Make sure digit is white on black background |
