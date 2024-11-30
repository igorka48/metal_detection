import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import kagglehub
import json

# Download dataset
path = kagglehub.dataset_download("fantacher/neu-metal-surface-defects-data")

# Check if dataset is downloaded successfully
if not os.path.exists(path):
    raise FileNotFoundError(f"Dataset not found at path: {path}")

# List the contents of the dataset directory to understand its structure
print("Dataset contents:", os.listdir(path))

# Define paths based on the actual dataset structure
# Dynamically search for training and validation directories
train_dir = None
validation_dir = None
for root, dirs, files in os.walk(path):
    if 'train' in dirs:
        train_dir = os.path.join(root, 'train')
    if 'valid' in dirs:
        validation_dir = os.path.join(root, 'valid')

# Verify if the paths exist
if train_dir is None:
    raise FileNotFoundError("Train directory not found in the dataset.")
if validation_dir is None:
    raise FileNotFoundError("Validation directory not found in the dataset.")

# Prepare data generators with augmentation for training and validation
data_gen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_data = data_gen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')

validation_gen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)
validation_data = validation_gen.flow_from_directory(validation_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')

# Define the model using MobileNetV2 (compatible with M1/M2/M3 Macs)
base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False

# Add custom layers
global_avg_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(train_data.num_classes, activation='softmax')

model = tf.keras.Sequential([
    base_model,
    global_avg_layer,
    prediction_layer
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_data, epochs=10, validation_data=validation_data)

# Save the trained model
model.save('metal_defect_model.keras')

# Save class indices to JSON
with open('class_indices.json', 'w') as json_file:
    json.dump(train_data.class_indices, json_file)

# Plot training history
import matplotlib.pyplot as plt

# Extract accuracy and loss values
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

# Plot accuracy
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')

plt.show()

# Function to test the model on a sample image
def predict_image(model, image_path, class_indices):
    # Load the image
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    input_data = np.expand_dims(tf.keras.preprocessing.image.img_to_array(img) / 255.0, axis=0)

    # Make prediction
    predictions = model.predict(input_data)

    # Map the predicted label indices to the class names
    class_labels = {v: k for k, v in class_indices.items()}

    # Print all predictions with their probabilities
    for idx, probability in enumerate(predictions[0]):
        if probability >= 0.1:
            label_name = class_labels[idx]
            print(f'{label_name}: {probability * 100:.2f}%')

