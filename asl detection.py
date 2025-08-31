import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.layers import (
    Conv2D, Dropout, Input, GlobalAveragePooling2D,
    Dense, BatchNormalization, MaxPool2D,
    RandomRotation, RandomZoom, RandomContrast,
    RandomBrightness, Rescaling
)

# -------------------------------
# Paths (keep your folder names)
# -------------------------------
path = "."  
train_path = os.path.join(path, "asl_alphabet_train")
test_path  = os.path.join(path, "asl_alphabet_test")

nested = os.path.join(train_path, "asl_alphabet_train")
if os.path.isdir(nested):
    train_path = nested  

# -------------------------------
# Train / Validation Datasets
# -------------------------------
BATCH_SIZE = 32
IMG_SIZE   = (64, 64)


train_data_raw = tf.keras.preprocessing.image_dataset_from_directory(
    train_path,
    validation_split=0.1,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_data_raw = tf.keras.preprocessing.image_dataset_from_directory(
    train_path,
    validation_split=0.1,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_data_raw.class_names  
print("Classes:", class_names)

# Optimized pipelines for training
AUTOTUNE  = tf.data.AUTOTUNE
train_data = train_data_raw.cache().shuffle(1000).prefetch(AUTOTUNE)
val_data   = val_data_raw.cache().prefetch(AUTOTUNE)

# -------------------------------
# Label Mapping for test parsing
# -------------------------------

m = {cls.lower(): idx for idx, cls in enumerate(class_names)}

if "del" in m and "delete" not in m:
    m["delete"] = m["del"]
if "delete" in m and "del" not in m:
    m["del"] = m["delete"]

# -------------------------------
# Load Test Data (flat folder)
# -------------------------------
test_images = []
test_labels = []

if os.path.isdir(test_path):
    for filename in os.listdir(test_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(test_path, filename)
        
            char = filename.split("_")[0].lower()
            if char in m:
                test_labels.append(m[char])
                img = load_img(img_path, target_size=IMG_SIZE)
                img_array = img_to_array(img)
                test_images.append(img_array)

test_data   = np.array(test_images, dtype="float32") / 255.0
test_labels = np.array(test_labels, dtype="int32")

print("Test Data Shape:", test_data.shape)
print("Test Labels Shape:", test_labels.shape)

# -------------------------------
# Preprocessing & Data Augmentation
# -------------------------------
preprocessing_and_data_augmentation = tf.keras.Sequential([
    Rescaling(1/255.),
    RandomRotation(0.07),
    RandomZoom(0.1),
    RandomContrast(0.1),
    RandomBrightness(0.1),
])

# -------------------------------
# Visualize Augmented Samples
# -------------------------------
images, labels = next(iter(train_data_raw))
fig, axis = plt.subplots(4, 4, figsize=(8, 8))
for i in range(4):
    for j in range(4):
        ax  = axis[i][j]
        img = images[4 * i + j]
        img = preprocessing_and_data_augmentation(tf.expand_dims(img, 0))[0]
        ax.imshow(tf.clip_by_value(img, 0.0, 1.0))
        ax.set_title(class_names[int(labels[4 * i + j])])
        ax.axis("off")
plt.tight_layout()
plt.show()

# -------------------------------
# Model Definition
# -------------------------------
NUMBER_OF_CLASSES = len(class_names)

model = tf.keras.models.Sequential([
    Input(shape=(64, 64, 3)),

    preprocessing_and_data_augmentation,

    Conv2D(32, (3, 3), padding="same", activation="relu"),
    BatchNormalization(),
    Dropout(0.2),
    MaxPool2D(),

    Conv2D(64, (3, 3), padding="same", activation="relu"),
    BatchNormalization(),
    Dropout(0.3),
    MaxPool2D(),

    Conv2D(128, (3, 3), padding="same", activation="relu"),
    BatchNormalization(),
    Dropout(0.3),
    MaxPool2D(),

    GlobalAveragePooling2D(),
    Dropout(0.3),
    Dense(128, activation="relu"),
    Dense(64, activation="relu"),
    Dense(NUMBER_OF_CLASSES, activation="softmax"),
])

# -------------------------------
# Compile & Train
# -------------------------------
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)
lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
)

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

history = model.fit(
    train_data,
    epochs=40,
    validation_data=val_data,
    callbacks=[early_stopping, lr_schedule]
)

# -------------------------------
# Evaluation (manual test folder)
# -------------------------------
if test_data.size > 0:
    test_loss, test_acc = model.evaluate(test_data, test_labels, verbose=2)
    print("Test Accuracy:", test_acc)
else:
    print("No test images found in:", test_path)






