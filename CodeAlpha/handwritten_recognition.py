# =============================================================
#   TASK 3: HANDWRITTEN CHARACTER RECOGNITION
#
#   What it does:
#       Looks at a handwritten digit image (28×28 pixels)
#       and predicts which digit (0–9) it is.
#
#   Dataset : MNIST — 70,000 handwritten digit images
#   Source  : Built into Keras — downloads automatically
#             (No Kaggle account needed)
#
#   Also supports EMNIST letters (A–Z) — see DATASET variable
#
#   Models used:
#       • Simple Neural Network (ANN) — baseline
#       • Convolutional Neural Network (CNN) — state-of-the-art
#
#   Author: Add your name here
# =============================================================


# ─────────────────────────────────────────────────────────────
# STEP 1 — IMPORT LIBRARIES
# ─────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
import os
warnings.filterwarnings("ignore")

# TensorFlow / Keras — for building neural networks
import tensorflow as tf
from tensorflow.keras.datasets  import mnist
from tensorflow.keras.models    import Sequential, Model
from tensorflow.keras.layers    import (
    Dense, Dropout, Flatten, BatchNormalization,
    Conv2D, MaxPooling2D, Input, GlobalAveragePooling2D
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from tensorflow.keras.utils               import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Scikit-learn — for evaluation metrics
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)

# Fix random seeds so results are reproducible
np.random.seed(42)
tf.random.set_seed(42)


# ─────────────────────────────────────────────────────────────
# CONFIGURATION — change these to switch between datasets
# ─────────────────────────────────────────────────────────────
DATASET    = "mnist"    # Options: "mnist" (digits 0-9) or "emnist" (letters A-Z)
EPOCHS     = 30         # Max training rounds (EarlyStopping stops earlier if needed)
BATCH_SIZE = 128        # Images processed per training step
IMG_SIZE   = 28         # MNIST images are 28×28 pixels


# ═══════════════════════════════════════════════════════════════
# STEP 2 — LOAD DATASET
#
#   What is MNIST?
#   ┌───────────────────────────────────────────────────────────┐
#   │ MNIST = Modified National Institute of Standards and Tech │
#   │ • 70,000 grayscale images of handwritten digits 0–9       │
#   │ • Each image is 28×28 pixels = 784 numbers (0–255)        │
#   │ • 60,000 training images + 10,000 test images             │
#   │ • Most famous benchmark dataset in deep learning          │
#   │ • Downloads automatically (~11 MB) on first run           │
#   └───────────────────────────────────────────────────────────┘
# ═══════════════════════════════════════════════════════════════
def load_dataset():
    print("\n" + "="*65)
    print("  STEP 2: LOADING DATASET")
    print("="*65)

    if DATASET == "emnist":
        # EMNIST = Extended MNIST (A–Z letters)
        # Install: pip install extra-keras-datasets
        try:
            from extra_keras_datasets import emnist
            (X_train, y_train), (X_test, y_test) = emnist.load_data(type="letters")
            y_train -= 1   # Labels are 1–26, shift to 0–25
            y_test  -= 1
            n_classes   = 26
            import string
            class_names = list(string.ascii_uppercase)
            print(f"  Dataset     : EMNIST Letters (A–Z)")
            print(f"  Classes     : 26 letters")
        except ImportError:
            print("  EMNIST requires: pip install extra-keras-datasets")
            print("  Falling back to MNIST digits...")
            return load_mnist_data()

    else:
        return load_mnist_data()

    print(f"  Train images: {X_train.shape[0]:,}")
    print(f"  Test images : {X_test.shape[0]:,}")
    print(f"  Image size  : {X_train.shape[1]}×{X_train.shape[2]} pixels")
    print(f"  Pixel range : {X_train.min()} to {X_train.max()}")
    return X_train, X_test, y_train, y_test, n_classes, class_names


def load_mnist_data():
    """Load the MNIST digits dataset."""
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    n_classes   = 10
    class_names = [str(i) for i in range(10)]

    print(f"  Dataset     : MNIST (Handwritten Digits 0–9)")
    print(f"  Classes     : 10 digits (0, 1, 2, ..., 9)")
    print(f"  Train images: {X_train.shape[0]:,}")
    print(f"  Test images : {X_test.shape[0]:,}")
    print(f"  Image size  : {X_train.shape[1]}×{X_train.shape[2]} pixels")
    print(f"  Pixel range : {X_train.min()} to {X_train.max()}")
    print(f"  Note        : Downloads ~11 MB on first run — wait a moment")

    return X_train, X_test, y_train, y_test, n_classes, class_names


# ═══════════════════════════════════════════════════════════════
# STEP 3 — EXPLORATORY DATA ANALYSIS
#
#   Before training any model, we LOOK at the data.
#   Understanding your data is the most important step in ML.
# ═══════════════════════════════════════════════════════════════
def explore_data(X_train, y_train, n_classes, class_names):
    print("\n" + "="*65)
    print("  STEP 3: EXPLORATORY DATA ANALYSIS (EDA)")
    print("="*65)

    # Count how many images per class
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"\n  Samples per class:")
    for cls, cnt in zip(unique, counts):
        bar = "█" * (cnt // 200)
        print(f"    {class_names[cls]} : {cnt:5d}  {bar}")

    # ── CHART 1: Sample images from each class ───────────────
    n_show = min(n_classes, 10)
    fig, axes = plt.subplots(2, n_show, figsize=(n_show * 1.8, 5))
    fig.suptitle("Sample Handwritten Images per Class",
                 fontsize=14, fontweight="bold")

    for col, cls in enumerate(range(n_show)):
        # Find first image of this class
        idx = np.where(y_train == cls)[0][0]

        # Top row: the image itself
        axes[0, col].imshow(X_train[idx], cmap="gray")
        axes[0, col].set_title(f"'{class_names[cls]}'", fontsize=12, fontweight="bold")
        axes[0, col].axis("off")

        # Bottom row: a different sample of same class
        idx2 = np.where(y_train == cls)[0][5]
        axes[1, col].imshow(X_train[idx2], cmap="gray")
        axes[1, col].axis("off")

    plt.tight_layout()
    plt.savefig("sample_images.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n  ✅ Sample images saved → sample_images.png")

    # ── CHART 2: Pixel intensity distribution ────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Pixel Analysis", fontsize=13, fontweight="bold")

    # Histogram of all pixel values
    axes[0].hist(X_train.flatten(), bins=50, color="#3498db", edgecolor="white")
    axes[0].set_title("Pixel Value Distribution\n(before normalization)")
    axes[0].set_xlabel("Pixel Value (0=Black, 255=White)")
    axes[0].set_ylabel("Count")
    axes[0].axvline(127, color="red", linestyle="--", label="Midpoint (127)")
    axes[0].legend()

    # Class distribution bar chart
    axes[1].bar(class_names[:n_classes], counts, color="#2ecc71", edgecolor="white")
    axes[1].set_title("Number of Images per Digit")
    axes[1].set_xlabel("Digit Class")
    axes[1].set_ylabel("Count")

    # Average image for each digit (what does an "average 3" look like?)
    avg_img = np.zeros((IMG_SIZE, n_show * IMG_SIZE))
    for i in range(n_show):
        class_imgs = X_train[y_train == i]
        avg = class_imgs.mean(axis=0)
        avg_img[:, i*IMG_SIZE:(i+1)*IMG_SIZE] = avg
    axes[2].imshow(avg_img, cmap="hot")
    axes[2].set_title("Average Image per Class\n(what each digit 'looks like' on average)")
    axes[2].set_xticks([IMG_SIZE//2 + i*IMG_SIZE for i in range(n_show)])
    axes[2].set_xticklabels(class_names[:n_show])
    axes[2].set_yticks([])

    plt.tight_layout()
    plt.savefig("pixel_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✅ Pixel analysis saved  → pixel_analysis.png")


# ═══════════════════════════════════════════════════════════════
# STEP 4 — PREPROCESS DATA
#
#   Raw images:  pixel values 0–255
#   After preprocessing: pixel values 0.0–1.0
#
#   WHY NORMALIZE?
#   Neural networks learn better when inputs are small numbers.
#   Dividing by 255 brings 0–255 into 0.0–1.0.
#   This is called "normalization" or "scaling".
#
#   WHY RESHAPE?
#   CNN expects shape: (samples, height, width, channels)
#   MNIST images are grayscale → channels = 1
#   So (60000, 28, 28) → (60000, 28, 28, 1)
# ═══════════════════════════════════════════════════════════════
def preprocess(X_train, X_test, y_train, y_test, n_classes):
    print("\n" + "="*65)
    print("  STEP 4: PREPROCESSING")
    print("="*65)

    print(f"\n  Before preprocessing:")
    print(f"    X_train shape : {X_train.shape}")
    print(f"    Pixel range   : {X_train.min()} – {X_train.max()}")
    print(f"    y_train sample: {y_train[:10]}  (raw integers)")

    # ── Normalize: 0–255 → 0.0–1.0 ──────────────────────────
    X_train_norm = X_train.astype("float32") / 255.0
    X_test_norm  = X_test.astype("float32")  / 255.0

    # ── Reshape: add channel dimension for CNN ────────────────
    # (samples, 28, 28) → (samples, 28, 28, 1)
    X_train_cnn  = X_train_norm[..., np.newaxis]   # adds a dimension at the end
    X_test_cnn   = X_test_norm[..., np.newaxis]

    # ── Flatten: for the simple ANN baseline ─────────────────
    # (samples, 28, 28, 1) → (samples, 784)
    # ANN takes a flat list of numbers, not a 2D image
    X_train_flat = X_train_norm.reshape(-1, IMG_SIZE * IMG_SIZE)
    X_test_flat  = X_test_norm.reshape(-1, IMG_SIZE * IMG_SIZE)

    # ── One-hot encode labels ─────────────────────────────────
    # "3" → [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    # "7" → [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    # This lets the model output a probability for each class.
    y_train_cat = to_categorical(y_train, n_classes)
    y_test_cat  = to_categorical(y_test,  n_classes)

    print(f"\n  After preprocessing:")
    print(f"    X_train_cnn shape : {X_train_cnn.shape}")
    print(f"    Pixel range       : {X_train_cnn.min():.1f} – {X_train_cnn.max():.1f}")
    print(f"    y_train_cat shape : {y_train_cat.shape}")
    print(f"    y_train_cat[0]    : {y_train_cat[0]}  "
          f"(one-hot for digit '{y_train[0]}')")
    print(f"\n  ✅ Preprocessing complete")

    return (X_train_cnn, X_test_cnn, X_train_flat, X_test_flat,
            y_train_cat, y_test_cat)


# ═══════════════════════════════════════════════════════════════
# STEP 5A — BUILD BASELINE MODEL (ANN)
#
#   ANN = Artificial Neural Network
#
#   Architecture:
#       Input (784) → Dense(256) → Dense(128) → Dense(64) → Output(10)
#
#   Analogy: Think of it as routing the image through layers of
#   "detectors", each layer learning more abstract patterns.
#   But ANN treats the image as a flat list — it ignores
#   the 2D structure of the pixels.
# ═══════════════════════════════════════════════════════════════
def build_ann(n_classes):
    model = Sequential([
        # Input layer — accepts 784 pixel values (28×28 flattened)
        Dense(256, activation="relu", input_shape=(IMG_SIZE * IMG_SIZE,)),
        # relu: Rectified Linear Unit — outputs 0 for negative, x for positive
        # Introduces non-linearity so the model can learn complex patterns
        BatchNormalization(),       # Normalizes layer outputs — faster training
        Dropout(0.3),               # Randomly turns off 30% of neurons — prevents overfitting

        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),

        Dense(64, activation="relu"),
        Dropout(0.2),

        # Output layer — 10 neurons, one per digit class
        # softmax: converts raw scores to probabilities that sum to 1.0
        Dense(n_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",                   # Adam = best general-purpose optimizer
        loss="categorical_crossentropy",    # Standard loss for multi-class classification
        metrics=["accuracy"]
    )

    print("\n  ANN Model Summary:")
    model.summary()
    return model


# ═══════════════════════════════════════════════════════════════
# STEP 5B — BUILD CNN MODEL
#
#   CNN = Convolutional Neural Network
#
#   WHY CNN IS BETTER THAN ANN FOR IMAGES:
#   • ANN sees pixels as a flat list → loses spatial information
#   • CNN looks at small regions (e.g. 3×3 patches) of the image
#     and learns patterns like edges, curves, loops → much better!
#
#   Architecture:
#       Conv Block 1 (32 filters)
#       → Conv Block 2 (64 filters)
#       → Conv Block 3 (128 filters)
#       → Dense Head → Output
#
#   What each part does:
#   ┌─────────────────────────────────────────────────────────────┐
#   │ Conv2D      — Scans the image with small filters (kernels). │
#   │               Detects edges, curves, corners etc.           │
#   │ BatchNorm   — Stabilizes training, speeds up convergence    │
#   │ MaxPooling  — Shrinks the image by keeping only the largest │
#   │               value in each 2×2 block. Reduces complexity.  │
#   │ Dropout     — Randomly turns off neurons. Prevents the      │
#   │               model from "memorising" training data.        │
#   │ Flatten     — Converts 2D feature map into a 1D vector      │
#   │ Dense       — Standard fully-connected layer for classifying│
#   └─────────────────────────────────────────────────────────────┘
# ═══════════════════════════════════════════════════════════════
def build_cnn(n_classes):
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 1))

    # ── Convolutional Block 1 ─────────────────────────────────
    # 32 filters scanning the image → detects simple features (edges, lines)
    x = Conv2D(32, kernel_size=(3,3), padding="same", activation="relu")(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(32, kernel_size=(3,3), padding="same", activation="relu")(x)
    x = MaxPooling2D(pool_size=(2,2))(x)   # 28×28 → 14×14
    x = Dropout(0.25)(x)

    # ── Convolutional Block 2 ─────────────────────────────────
    # 64 filters → detects more complex features (curves, loops)
    x = Conv2D(64, kernel_size=(3,3), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, kernel_size=(3,3), padding="same", activation="relu")(x)
    x = MaxPooling2D(pool_size=(2,2))(x)   # 14×14 → 7×7
    x = Dropout(0.25)(x)

    # ── Convolutional Block 3 ─────────────────────────────────
    # 128 filters → detects high-level shapes
    x = Conv2D(128, kernel_size=(3,3), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)   # 7×7 → 3×3
    x = Dropout(0.3)(x)

    # ── Classification Head ───────────────────────────────────
    x = Flatten()(x)                       # 3×3×128 = 1152 numbers
    x = Dense(512, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)                    # Aggressive dropout before final layer
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.4)(x)
    outputs = Dense(n_classes, activation="softmax")(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("\n  CNN Model Summary:")
    model.summary()
    return model


# ═══════════════════════════════════════════════════════════════
# STEP 6 — DATA AUGMENTATION
#
#   What is data augmentation?
#   We apply small random transformations to training images.
#   This makes the model see MORE variety → less overfitting.
#
#   Transforms applied:
#     • Rotation     ± 10° — digits can be slightly tilted
#     • Width shift  10%   — digit can be slightly left/right
#     • Height shift 10%   — digit can be slightly up/down
#     • Zoom         10%   — digit can be slightly bigger/smaller
#
#   NOTE: We ONLY augment TRAINING data, NEVER the test data.
#   The test set must stay unchanged — it represents real-world images.
# ═══════════════════════════════════════════════════════════════
def get_augmentation():
    return ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.10,
        height_shift_range=0.10,
        zoom_range=0.10,
        shear_range=0.10,
    )


# ═══════════════════════════════════════════════════════════════
# STEP 7 — TRAIN MODELS
# ═══════════════════════════════════════════════════════════════
def train_ann(model, X_train_flat, X_test_flat, y_train_cat, y_test_cat):
    print("\n" + "="*65)
    print("  STEP 7A: TRAINING ANN (Simple Neural Network)")
    print("="*65)

    callbacks = [
        EarlyStopping(patience=7, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(patience=4, factor=0.5, min_lr=1e-6, verbose=1),
    ]

    history = model.fit(
        X_train_flat, y_train_cat,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test_flat, y_test_cat),
        callbacks=callbacks,
        verbose=1
    )
    print("\n  ✅ ANN training complete")
    return history


def train_cnn(model, X_train_cnn, X_test_cnn, y_train_cat, y_test_cat):
    print("\n" + "="*65)
    print("  STEP 7B: TRAINING CNN (Convolutional Neural Network)")
    print("="*65)
    print("  Using data augmentation to prevent overfitting...")

    datagen = get_augmentation()
    datagen.fit(X_train_cnn)

    callbacks = [
        EarlyStopping(patience=8, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(patience=4, factor=0.5, min_lr=1e-7, verbose=1),
        ModelCheckpoint("best_cnn.keras", save_best_only=True, verbose=0),
    ]

    history = model.fit(
        datagen.flow(X_train_cnn, y_train_cat, batch_size=BATCH_SIZE),
        steps_per_epoch=len(X_train_cnn) // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test_cnn, y_test_cat),
        callbacks=callbacks,
        verbose=1
    )
    print("\n  ✅ CNN training complete")
    return history


# ═══════════════════════════════════════════════════════════════
# STEP 8 — EVALUATE & VISUALIZE RESULTS
# ═══════════════════════════════════════════════════════════════
def evaluate_and_plot(ann_model, cnn_model,
                       ann_history, cnn_history,
                       X_test_flat, X_test_cnn, X_test_raw,
                       y_test_cat, y_test,
                       n_classes, class_names):
    print("\n" + "="*65)
    print("  STEP 8: EVALUATION & RESULTS")
    print("="*65)

    # ── Get predictions ───────────────────────────────────────
    ann_pred_prob = ann_model.predict(X_test_flat, verbose=0)
    cnn_pred_prob = cnn_model.predict(X_test_cnn,  verbose=0)
    ann_pred      = np.argmax(ann_pred_prob, axis=1)
    cnn_pred      = np.argmax(cnn_pred_prob, axis=1)

    ann_acc  = accuracy_score(y_test, ann_pred)
    cnn_acc  = accuracy_score(y_test, cnn_pred)
    _, ann_val_acc = ann_model.evaluate(X_test_flat, y_test_cat, verbose=0)
    _, cnn_val_acc = cnn_model.evaluate(X_test_cnn,  y_test_cat, verbose=0)

    print(f"\n  {'Model':<30} {'Test Accuracy':>15}  {'Test Error Rate':>15}")
    print(f"  {'─'*60}")
    print(f"  {'ANN (Simple Neural Network)':<30} {ann_acc*100:>14.2f}%  "
          f"{(1-ann_acc)*100:>14.2f}%")
    print(f"  {'CNN (Convolutional Neural Net)':<30} {cnn_acc*100:>14.2f}%  "
          f"{(1-cnn_acc)*100:>14.2f}%")
    print(f"\n  CNN correctly classified "
          f"{int(cnn_acc * len(y_test)):,} out of {len(y_test):,} test images")
    print(f"  CNN only made {int((1-cnn_acc)*len(y_test))} mistakes out of {len(y_test):,}!")

    print(f"\n  Detailed CNN Classification Report:")
    print(classification_report(y_test, cnn_pred, target_names=class_names))

    # ── DASHBOARD: 6-panel evaluation figure ─────────────────
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle("Handwritten Character Recognition — Full Evaluation Dashboard",
                 fontsize=15, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)

    # ── Panel 1: Training Accuracy ────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(ann_history.history["accuracy"],     lw=2, color="#e67e22", label="ANN Train")
    ax1.plot(ann_history.history["val_accuracy"], lw=2, color="#e67e22",
             linestyle="--", label="ANN Val")
    ax1.plot(cnn_history.history["accuracy"],     lw=2, color="#3498db", label="CNN Train")
    ax1.plot(cnn_history.history["val_accuracy"], lw=2, color="#3498db",
             linestyle="--", label="CNN Val")
    ax1.set_title("Training Accuracy (ANN vs CNN)")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
    ax1.set_ylim(0.8, 1.01)

    # ── Panel 2: Training Loss ────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(ann_history.history["loss"],     lw=2, color="#e67e22", label="ANN Train Loss")
    ax2.plot(ann_history.history["val_loss"], lw=2, color="#e67e22",
             linestyle="--", label="ANN Val Loss")
    ax2.plot(cnn_history.history["loss"],     lw=2, color="#3498db", label="CNN Train Loss")
    ax2.plot(cnn_history.history["val_loss"], lw=2, color="#3498db",
             linestyle="--", label="CNN Val Loss")
    ax2.set_title("Training Loss (lower = better)")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    # ── Panel 3: ANN vs CNN Accuracy bar ─────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    models_names = ["ANN\n(Simple)", "CNN\n(Convolutional)"]
    accs         = [ann_acc * 100, cnn_acc * 100]
    bars = ax3.bar(models_names, accs, color=["#e67e22", "#3498db"],
                   edgecolor="white", linewidth=1.5, width=0.5)
    ax3.set_ylim(90, 100)
    ax3.set_title("Final Test Accuracy Comparison")
    ax3.set_ylabel("Accuracy (%)")
    for bar, acc in zip(bars, accs):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f"{acc:.2f}%", ha="center", fontweight="bold", fontsize=11)
    ax3.axhline(99, color="green", linestyle="--", lw=1, alpha=0.6, label="99% line")
    ax3.legend(fontsize=8)

    # ── Panel 4: CNN Confusion Matrix ────────────────────────
    ax4 = fig.add_subplot(gs[1, :2])
    cm  = confusion_matrix(y_test, cnn_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.3, ax=ax4, cbar_kws={"shrink": 0.8})
    ax4.set_title("CNN Confusion Matrix\n(Rows = Actual, Columns = Predicted)")
    ax4.set_xlabel("Predicted Digit")
    ax4.set_ylabel("Actual Digit")

    # ── Panel 5: Per-class accuracy ───────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    per_class_acc = []
    for cls in range(n_classes):
        mask    = y_test == cls
        cls_acc = accuracy_score(y_test[mask], cnn_pred[mask])
        per_class_acc.append(cls_acc * 100)
    colors_cls = ["#2ecc71" if a >= 99 else "#e67e22" if a >= 97 else "#e74c3c"
                  for a in per_class_acc]
    ax5.barh(class_names, per_class_acc, color=colors_cls)
    ax5.set_xlim(90, 100.5)
    ax5.set_title("Per-Class Accuracy (CNN)\nGreen≥99%, Orange≥97%, Red<97%")
    ax5.set_xlabel("Accuracy (%)")
    for i, v in enumerate(per_class_acc):
        ax5.text(v + 0.05, i, f"{v:.1f}%", va="center", fontsize=8)
    ax5.axvline(99, color="green", linestyle="--", lw=1, alpha=0.5)

    # ── Panel 6: Show correct predictions ────────────────────
    ax6 = fig.add_subplot(gs[2, :2])
    ax6.axis("off")
    correct_idx = np.where(cnn_pred == y_test)[0][:12]
    for i, idx in enumerate(correct_idx):
        sub = ax6.inset_axes([i/12, 0.1, 1/12-0.01, 0.8])
        sub.imshow(X_test_raw[idx], cmap="gray")
        sub.set_title(f"✓ {class_names[cnn_pred[idx]]}", fontsize=9,
                      color="green", fontweight="bold")
        sub.axis("off")
    ax6.set_title("Sample CORRECT Predictions (CNN)", fontweight="bold", x=0.5, y=1.0)

    # ── Panel 7: Show wrong predictions ──────────────────────
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis("off")
    wrong_idx = np.where(cnn_pred != y_test)[0][:6]
    for i, idx in enumerate(wrong_idx[:6]):
        row = i // 3; col = i % 3
        sub = ax7.inset_axes([col/3, (1-row)*0.5, 1/3-0.02, 0.48])
        sub.imshow(X_test_raw[idx], cmap="Reds")
        sub.set_title(f"True:{class_names[y_test[idx]]}\nPred:{class_names[cnn_pred[idx]]}",
                      fontsize=7, color="red")
        sub.axis("off")
    ax7.set_title("❌ Mistakes (CNN)", fontweight="bold")

    plt.savefig("evaluation_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n  ✅ Full dashboard saved → evaluation_dashboard.png")

    return ann_acc, cnn_acc


# ═══════════════════════════════════════════════════════════════
# STEP 9 — VISUALIZE CNN FILTERS
#
#   What do the CNN filters "see"?
#   After training, we can look at what patterns each filter
#   has learned to detect. Early filters detect simple things
#   like horizontal lines, vertical lines, diagonal edges.
# ═══════════════════════════════════════════════════════════════
def visualize_filters(cnn_model):
    print("\n[STEP 9] Visualizing CNN Learned Filters...")

    # Get weights from first convolutional layer
    first_conv_weights = cnn_model.layers[1].get_weights()[0]
    # Shape: (3, 3, 1, 32) → 32 filters, each 3×3
    n_filters = first_conv_weights.shape[-1]
    n_show    = min(n_filters, 32)

    fig, axes = plt.subplots(4, 8, figsize=(14, 8))
    fig.suptitle(f"CNN Learned Filters (Layer 1 — {n_show} filters)\n"
                 "Each filter detects a specific visual pattern (edge, curve, etc.)",
                 fontsize=12, fontweight="bold")

    for i, ax in enumerate(axes.flatten()):
        if i < n_show:
            f = first_conv_weights[:, :, 0, i]
            # Normalize to 0–1 for display
            f = (f - f.min()) / (f.max() - f.min() + 1e-8)
            ax.imshow(f, cmap="viridis")
            ax.set_title(f"F{i+1}", fontsize=7)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("cnn_filters.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✅ Filter visualization saved → cnn_filters.png")


# ═══════════════════════════════════════════════════════════════
# STEP 10 — PREDICT A CUSTOM IMAGE
#
#   Draw a digit yourself using Windows Paint / Preview (Mac)
#   Save as PNG (28×28 pixels, black digit on white background)
#   Then run: predict_custom_image("your_digit.png", cnn_model)
# ═══════════════════════════════════════════════════════════════
def predict_custom_image(image_path, model, class_names):
    """
    Predict digit from your own handwritten image file.

    How to create your own image:
    1. Open Paint (Windows) or Preview (Mac)
    2. Create a new image: 28×28 pixels
    3. Fill background with BLACK (important!)
    4. Draw your digit in WHITE
    5. Save as .png
    6. Call: predict_custom_image("my_digit.png", cnn_model, class_names)
    """
    try:
        from PIL import Image
        img = Image.open(image_path).convert("L")  # Convert to grayscale
        img = img.resize((IMG_SIZE, IMG_SIZE))      # Resize to 28×28
        arr = np.array(img).astype("float32") / 255.0
        arr = arr.reshape(1, IMG_SIZE, IMG_SIZE, 1)

        probs   = model.predict(arr, verbose=0)[0]
        pred    = np.argmax(probs)
        conf    = probs[pred] * 100

        print(f"\n  📷 Custom Image: {image_path}")
        print(f"  ⚑  Predicted Digit : {class_names[pred]}")
        print(f"  ⚑  Confidence      : {conf:.1f}%")
        print(f"\n  Full probability distribution:")
        for i, (cls, p) in enumerate(zip(class_names, probs)):
            bar = "█" * int(p * 40)
            print(f"    {cls} : {bar:<40} {p*100:5.1f}%")
        return pred

    except ImportError:
        print("  PIL not installed. Run: pip install Pillow")
    except FileNotFoundError:
        print(f"  File '{image_path}' not found.")


# ═══════════════════════════════════════════════════════════════
# STEP 10B — DEMO PREDICTIONS (no image file needed)
#
#   Take real images from the test set and show predictions.
#   This proves the model works without needing custom images.
# ═══════════════════════════════════════════════════════════════
def demo_predictions(cnn_model, X_test_cnn, X_test_raw, y_test, class_names):
    print("\n" + "="*65)
    print("  STEP 10: DEMO PREDICTIONS (Real Test Images)")
    print("="*65)

    # Pick one sample of each digit class
    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    fig.suptitle("Model Predictions on Real Test Images",
                 fontsize=13, fontweight="bold")

    for cls in range(10):
        row = cls // 5; col = cls % 5
        idx = np.where(y_test == cls)[0][0]

        img   = X_test_cnn[idx]
        probs = cnn_model.predict(img[np.newaxis], verbose=0)[0]
        pred  = np.argmax(probs)
        conf  = probs[pred] * 100

        axes[row, col].imshow(X_test_raw[idx], cmap="gray")
        color = "green" if pred == cls else "red"
        axes[row, col].set_title(
            f"Actual: {class_names[cls]}\n"
            f"Pred: {class_names[pred]} ({conf:.0f}%)",
            color=color, fontsize=9, fontweight="bold"
        )
        axes[row, col].axis("off")

        correct = "✅" if pred == cls else "❌"
        print(f"  Digit {class_names[cls]} → Predicted: {class_names[pred]}  "
              f"Confidence: {conf:.1f}%  {correct}")

    plt.tight_layout()
    plt.savefig("demo_predictions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n  ✅ Demo predictions saved → demo_predictions.png")


# ─────────────────────────────────────────────────────────────
# MAIN — Run everything in order
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("="*65)
    print("   TASK 3: HANDWRITTEN CHARACTER RECOGNITION")
    print("   Dataset: MNIST | Models: ANN + CNN")
    print("="*65)

    # ── Run all steps ─────────────────────────────────────────
    X_train, X_test, y_train, y_test, n_classes, class_names = load_dataset()

    explore_data(X_train, y_train, n_classes, class_names)

    (X_train_cnn, X_test_cnn,
     X_train_flat, X_test_flat,
     y_train_cat, y_test_cat) = preprocess(X_train, X_test, y_train, y_test, n_classes)

    # Build both models
    ann_model = build_ann(n_classes)
    cnn_model = build_cnn(n_classes)

    # Train both
    ann_history = train_ann(ann_model, X_train_flat, X_test_flat,
                            y_train_cat, y_test_cat)
    cnn_history = train_cnn(cnn_model, X_train_cnn, X_test_cnn,
                            y_train_cat, y_test_cat)

    # Evaluate and visualize
    ann_acc, cnn_acc = evaluate_and_plot(
        ann_model, cnn_model,
        ann_history, cnn_history,
        X_test_flat, X_test_cnn, X_test,
        y_test_cat, y_test,
        n_classes, class_names
    )

    visualize_filters(cnn_model)
    demo_predictions(cnn_model, X_test_cnn, X_test, y_test, class_names)

    # Save final model
    cnn_model.save("handwriting_cnn_model.keras")
    print(f"\n  ✅ CNN model saved → handwriting_cnn_model.keras")

    # ── To predict your own image (draw and save as PNG first) ─
    # predict_custom_image("my_digit.png", cnn_model, class_names)

    print("\n" + "="*65)
    print("  ✅  TASK 3 COMPLETE!")
    print("  Files created:")
    print("    sample_images.png        — One sample per digit class")
    print("    pixel_analysis.png       — Pixel value distributions")
    print("    evaluation_dashboard.png — Full 6-panel evaluation")
    print("    cnn_filters.png          — Learned CNN filter patterns")
    print("    demo_predictions.png     — Live predictions on test images")
    print("    handwriting_cnn_model.keras — Saved trained model")
    print("="*65)
    print(f"\n  ANN Accuracy : {ann_acc*100:.2f}%")
    print(f"  CNN Accuracy : {cnn_acc*100:.2f}%  ← State-of-the-art!")
