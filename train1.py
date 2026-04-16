import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Paths
DATA_DIR = r"D:\project2\aerial project\dataset"
MODEL_PATH = r"D:\project2\aerial project\bird_drone_model.keras"

# Settings
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 123

# Load dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

print("Class names:", train_ds.class_names)

# Performance improvement
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Data augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# Base model
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

# Build model
inputs = layers.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = models.Model(inputs, outputs)

# Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Callbacks
early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

# Train
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=[early_stop, checkpoint]
)

# Save final model
model.save(MODEL_PATH)
print(f"Model saved at: {MODEL_PATH}")


import matplotlib.pyplot as plt

# 1 Accuracy
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy')
plt.legend(['Train','Val'])
plt.savefig('1_accuracy.png')

# 2 Loss
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss')
plt.legend(['Train','Val'])
plt.savefig('2_loss.png')

# 3 Accuracy difference
plt.figure()
plt.plot([t-v for t,v in zip(history.history['accuracy'], history.history['val_accuracy'])])
plt.title('Accuracy Gap')
plt.savefig('3_acc_gap.png')

# 4 Loss difference
plt.figure()
plt.plot([t-v for t,v in zip(history.history['loss'], history.history['val_loss'])])
plt.title('Loss Gap')
plt.savefig('4_loss_gap.png')

# 5 Accuracy bar
plt.figure()
plt.bar(['Train','Val'], [max(history.history['accuracy']), max(history.history['val_accuracy'])])
plt.title('Max Accuracy')
plt.savefig('5_acc_bar.png')

# 6 Loss bar
plt.figure()
plt.bar(['Train','Val'], [min(history.history['loss']), min(history.history['val_loss'])])
plt.title('Min Loss')
plt.savefig('6_loss_bar.png')

# 7 Epoch vs Accuracy
plt.figure()
plt.scatter(range(len(history.history['accuracy'])), history.history['accuracy'])
plt.title('Epoch vs Accuracy')
plt.savefig('7_epoch_acc.png')

# 8 Epoch vs Loss
plt.figure()
plt.scatter(range(len(history.history['loss'])), history.history['loss'])
plt.title('Epoch vs Loss')
plt.savefig('8_epoch_loss.png')

# 9 Histogram Accuracy
plt.figure()
plt.hist(history.history['accuracy'])
plt.title('Accuracy Distribution')
plt.savefig('9_hist_acc.png')

# 10 Histogram Loss
plt.figure()
plt.hist(history.history['loss'])
plt.title('Loss Distribution')
plt.savefig('10_hist_loss.png')

# 11 Validation Accuracy Trend
plt.figure()
plt.plot(history.history['val_accuracy'], marker='o')
plt.title('Validation Accuracy Trend')
plt.savefig('11_val_trend.png')

# 12 Validation Loss Trend
plt.figure()
plt.plot(history.history['val_loss'], marker='o')
plt.title('Validation Loss Trend')
plt.savefig('12_val_loss_trend.png')