import os

base_path = "/content/drive/MyDrive/ML"

for root, dirs, files in os.walk(base_path):
    for file in files:
        if file.endswith(".zip"):
            print(os.path.join(root, file))
import os
os.listdir('/content/drive/MyDrive/ML/')
from google.colab import drive
drive.mount('/content/drive')
import os
os.listdir('/content/drive/MyDrive/ML/')
import zipfile
import os
from google.colab import drive

# 1. Drive is already mounted, so we don't need to mount it again.
# if not os.path.exists('/content/drive'):
#     drive.mount('/content/drive')

# 2. Use the FULL ABSOLUTE PATH (starts with /content/)
zip_path = '/content/drive/MyDrive/ML/RobberyVideoDataset.zip'
extract_path = '/content/drive/MyDrive/ML/unzipped_data'

# 3. Double check if the file exists before unzipping
if os.path.exists(zip_path):
    print("Found the zip file! Unzipping now...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print(f"Done! Your files are at: {extract_path}")
else:
    print(f"STILL NOT FOUND: I looked at {zip_path} and saw nothing.")
    print("Check if 'ML' is capitalized or if the filename is slightly different.")


import os

os.listdir('/content/drive/MyDrive/ML/unzipped_data')
import os

os.listdir('/content/drive/MyDrive/')

import os

base_path = '/content/drive/MyDrive/ML/unzipped_data'

print("Normal samples:", os.listdir(base_path + '/normal')[:5])
print("Shoplifting samples:", os.listdir(base_path + '/shoplifting')[:5])
import cv2
import os
import time

# paths
base_path = '/content/drive/MyDrive/ML/unzipped_data'
output_path = '/content/drive/MyDrive/ML/frames'

os.makedirs(output_path, exist_ok=True)

classes = ['normal', 'shoplifting']

frame_skip = 30   # take 1 frame every ~1 sec (assuming ~30 fps)

start_time = time.time() # Start timer

for label in classes:
    video_folder = os.path.join(base_path, label)
    save_folder = os.path.join(output_path, label)
    os.makedirs(save_folder, exist_ok=True)

    for video_name in os.listdir(video_folder):
        video_path = os.path.join(video_folder, video_name)

        cap = cv2.VideoCapture(video_path)
        count = 0
        frame_id = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if count % frame_skip == 0:
                frame_filename = f"{video_name}_frame{frame_id}.jpg"
                cv2.imwrite(os.path.join(save_folder, frame_filename), frame)
                frame_id += 1

            count += 1

        cap.release()

end_time = time.time() # End timer
elapsed_time = end_time - start_time

print("Frame extraction done")
print(f"Time taken for frame extraction: {elapsed_time:.2f} seconds")
import tensorflow as tf

# Define parameters
IMG_SIZE = 128
BATCH_SIZE = 32 # You can adjust this batch size
base_path = '/content/drive/MyDrive/ML/frames'

# Create a dataset for training and validation
# We'll let TensorFlow handle the splitting and resizing.
# Note: This will load images at the specified IMG_SIZE directly from disk.
# You might want to remove the redundant resizing in cell `2W0-Lt3z0ZF3` if you keep that cell for other purposes,
# or consider saving the frames at the target IMG_SIZE during extraction.

# Using a validation_split directly creates train and validation datasets
train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
    base_path,
    labels='inferred',
    label_mode='binary', # For binary classification
    image_size=(IMG_SIZE, IMG_SIZE),
    interpolation='bilinear',
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42,
    validation_split=0.2,
    subset='both'
)

# Normalize pixel values to [0, 1] range
def normalize_img(image, label):
    return image / 255.0, label

train_ds = train_ds.map(normalize_img)
val_ds = val_ds.map(normalize_img)

print("Train Dataset element spec:", train_ds.element_spec)
print("Validation Dataset element spec:", val_ds.element_spec)

# Prefetching data for performance
train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model_optimized = Sequential()

# Conv Block 1
model_optimized.add(Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)))
model_optimized.add(MaxPooling2D(2,2))

# Conv Block 2
model_optimized.add(Conv2D(64, (3,3), activation='relu'))
model_optimized.add(MaxPooling2D(2,2))

# Conv Block 3
model_optimized.add(Conv2D(128, (3,3), activation='relu'))
model_optimized.add(MaxPooling2D(2,2))

# Flatten
model_optimized.add(Flatten())

# Dense layers
model_optimized.add(Dense(128, activation='relu'))
model_optimized.add(Dropout(0.5))

# Output layer (binary classification)
model_optimized.add(Dense(1, activation='sigmoid'))

# Compile
model_optimized.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model_optimized.summary()
# Train the model using the new datasets
history_optimized = model_optimized.fit(
    train_ds,
    epochs=10, # You can adjust the number of epochs
    validation_data=val_ds
)
import os

print("Normal frames:", os.listdir('/content/drive/MyDrive/ML/frames/normal')[:5])
print("Shoplifting frames:", os.listdir('/content/drive/MyDrive/ML/frames/shoplifting')[:5])
import os
import cv2
import numpy as np

# paths
base_path = '/content/drive/MyDrive/ML/frames'

IMG_SIZE = 128  # keep small to avoid crash

X = []
y = []

classes = {
    'normal': 0,
    'shoplifting': 1
}

for label, class_id in classes.items():
    folder = os.path.join(base_path, label)

    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)

        img = cv2.imread(img_path)

        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        X.append(img)
        y.append(class_id)

# convert to numpy
X = np.array(X)
y = np.array(y)

print("X shape:", X.shape)
print("y shape:", y.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
X_train = X_train / 255.0
X_test = X_test / 255.0

print("Min value:", X_train.min())
print("Max value:", X_train.max())
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential()

# Conv Block 1
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)))
model.add(MaxPooling2D(2,2))

# Conv Block 2
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

# Conv Block 3
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

# Flatten
model.add(Flatten())

# Dense layers
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

# Output layer (binary classification)
model.add(Dense(1, activation='sigmoid'))

# Compile
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

loss, accuracy = model.evaluate(X_test, y_test)

print("Test Loss:", loss)
print("Test Accuracy:", accuracy)
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# predictions
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int).reshape(-1)

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# classification report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


y_pred = (y_pred > 0.3).astype(int)
model_optimized.save('/content/drive/MyDrive/ML/shoplifting_model_optimized.h5')
print("Optimized model saved")
from tensorflow.keras.models import load_model
import cv2
import numpy as np

# load model
model = load_model('/content/drive/MyDrive/ML/shoplifting_model_optimized.h5')

# Re-compile the model after loading
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# load one image (CHANGE PATH)
img_path = '/content/drive/MyDrive/ML/frames/shoplifting/shoplifting-1.mp4_frame0.jpg'

img = cv2.imread(img_path)
img = cv2.resize(img, (128,128))
img = img / 255.0

img = np.expand_dims(img, axis=0)

# prediction
pred = model.predict(img)[0][0]

print("Raw prediction:", pred)

if pred > 0.5:
    print("Prediction: SHOPLIFTING")
else:
    print("Prediction: NORMAL")
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# load model once
model = load_model('/content/drive/MyDrive/ML/shoplifting_model_optimized.h5')
IMG_SIZE = 128

def predict_image(img_path):
    img = cv2.imread(img_path)

    if img is None:
        return "Error: Image not found"

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0][0]

    if pred > 0.5:
        return f"SHOPLIFTING ({pred:.4f})"
    else:
        return f"NORMAL ({pred:.4f})"
%%writefile predict.py
import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('/content/drive/MyDrive/ML/shoplifting_model_optimized.h5')

IMG_SIZE = 128

def predict_image(img_path):
    img = cv2.imread(img_path)

    if img is None:
        return "Error: Image not found"

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0][0]

    if pred > 0.5:
        return f"SHOPLIFTING ({pred:.4f})"
    else:
        return f"NORMAL ({pred:.4f})"
from predict import predict_image
result = predict_image('/content/drive/MyDrive/ML/frames/shoplifting/shoplifting-1.mp4_frame0.jpg')
print(result)
import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('/content/drive/MyDrive/ML/shoplifting_model_optimized.h5')

IMG_SIZE = 128

def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)

    frame_skip = 30
    count = 0

    predictions = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_skip == 0:
            img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0
            img = np.expand_dims(img, axis=0)

            pred = model.predict(img)[0][0]
            predictions.append(pred)

        count += 1

    cap.release()

    avg_pred = np.mean(predictions)

    print("Average prediction:", avg_pred)

    if avg_pred > 0.5:
        print("FINAL RESULT: SHOPLIFTING")
    else:
        print("FINAL RESULT: NORMAL")
%%writefile app.py
import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tempfile

model = load_model('/content/drive/MyDrive/ML/shoplifting_model_optimized.h5')

IMG_SIZE = 128

st.title("Shoplifting Detection App")

uploaded_file = st.file_uploader("Upload a video", type=["mp4"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)

    frame_skip = 30
    count = 0
    predictions = []

    st.write("Processing video...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_skip == 0:
            img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0
            img = np.expand_dims(img, axis=0)

            pred = model.predict(img)[0][0]
            predictions.append(pred)

        count += 1

    cap.release()

    avg_pred = np.mean(predictions)

    st.write(f"Average prediction: {avg_pred:.4f}")

    if avg_pred > 0.5:
        st.error("SHOPLIFTING DETECTED 🚨")
    else:
        st.success("NORMAL ACTIVITY ✅")
from google.colab import files

files.download('/content/app.py')
files.download('/content/drive/MyDrive/ML/shoplifting_model_optimized.h5')

history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test)
)
import matplotlib.pyplot as plt

# Plot training & validation loss values
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss over Epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()
import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy over Epochs')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'cm' (confusion matrix) and 'classes' are already defined
# cm = np.array([[TN, FP], [FN, TP]]) from cell GmGAZPlA26jg
# classes = ['normal', 'shoplifting'] from previous cells

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Normal', 'Predicted Shoplifting'],
            yticklabels=['Actual Normal', 'Actual Shoplifting'])
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()
!pip install -q kaggle decord transformers torchvision pytorchvideo
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p ~/.kaggle
!cp /content/drive/MyDrive/kaggle/kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!ls /content/drive/MyDrive/ML/
import zipfile

zip_path = "/content/drive/MyDrive/ML/RobberyVideoDataset.zip"
extract_path = "/content/dataset"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print("Dataset extracted successfully!")
import os

for root, dirs, files in os.walk("/content/dataset"):
    print("ROOT:", root)
    print("FOLDERS:", dirs)
    print("FILES:", files[:5])
    break
import torch
from torch.utils.data import Dataset
from decord import VideoReader, cpu
import os

class VideoDataset(Dataset):
    def __init__(self, root_dir, num_frames=16):
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.samples = []
        self.labels = []

        classes = sorted(os.listdir(root_dir))  # important
        self.class_to_idx = {cls: i for i, cls in enumerate(classes)}

        for cls in classes:
            cls_path = os.path.join(root_dir, cls)
            for file in os.listdir(cls_path):
                if file.endswith((".mp4", ".avi", ".mov")):
                    self.samples.append(os.path.join(cls_path, file))
                    self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path = self.samples[idx]
        label = self.labels[idx]

        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)

            indices = torch.linspace(0, total_frames - 1, self.num_frames).long()
            frames = vr.get_batch(indices).asnumpy()

            frames = torch.tensor(frames).permute(3, 0, 1, 2)  # C T H W
            frames = frames.float() / 255.0

        except Exception as e:
            print("Error loading:", video_path)
            frames = torch.zeros((3, self.num_frames, 224, 224))

        return frames, label
from torch.utils.data import DataLoader

dataset = VideoDataset("/content/dataset")

print("Total videos:", len(dataset))

loader = DataLoader(dataset, batch_size=2, shuffle=True)
for videos, labels in loader:
    print("Video shape:", videos.shape)
    print("Labels:", labels)
    break
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from decord import VideoReader, cpu
import os

class VideoDatasetV2(Dataset):   # ✅ NEW NAME
    def __init__(self, root_dir, num_frames=16):
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.samples = []
        self.labels = []

        classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(classes)}

        for cls in classes:
            cls_path = os.path.join(root_dir, cls)
            for file in os.listdir(cls_path):
                if file.endswith((".mp4", ".avi", ".mov")):
                    self.samples.append(os.path.join(cls_path, file))
                    self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path = self.samples[idx]
        label = self.labels[idx]

        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)

        indices = torch.linspace(0, total_frames - 1, self.num_frames).long()
        frames = vr.get_batch(indices).asnumpy()

        frames = torch.tensor(frames).permute(3, 0, 1, 2)  # C T H W
        frames = frames.float() / 255.0

        # ✅ GUARANTEED RESIZE FIX
        frames = F.interpolate(frames, size=(224, 224), mode='bilinear', align_corners=False)

        return frames, label
from torch.utils.data import DataLoader

dataset = VideoDatasetV2("/content/dataset")
loader = DataLoader(dataset, batch_size=2, shuffle=True)
for videos, labels in loader:
    print("Video shape:", videos.shape)
    print("Labels:", labels)
    break
from transformers import VideoMAEForVideoClassification

model = VideoMAEForVideoClassification.from_pretrained(
    "MCG-NJU/videomae-base-finetuned-kinetics",
    num_labels=2,
    ignore_mismatched_sizes=True   # 🔥 IMPORTANT FIX
)
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("Using device:", device)
from transformers import VideoMAEForVideoClassification

model = VideoMAEForVideoClassification.from_pretrained(
    "MCG-NJU/videomae-base-finetuned-kinetics",
    num_labels=2,
    ignore_mismatched_sizes=True   # 🔥 IMPORTANT FIX
)
loader = DataLoader(dataset, batch_size=1, shuffle=True)
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from decord import VideoReader, cpu
import os

class VideoDatasetV3(Dataset):
    def __init__(self, root_dir, num_frames=16):
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.samples = []
        self.labels = []

        classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(classes)}

        for cls in classes:
            cls_path = os.path.join(root_dir, cls)
            for file in os.listdir(cls_path):
                if file.endswith((".mp4", ".avi", ".mov")):
                    self.samples.append(os.path.join(cls_path, file))
                    self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path = self.samples[idx]
        label = self.labels[idx]

        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)

        indices = torch.linspace(0, total_frames - 1, self.num_frames).long()
        frames = vr.get_batch(indices).asnumpy()

        # ✅ FIXED FORMAT
        frames = torch.tensor(frames).permute(0, 3, 1, 2)  # T C H W
        frames = frames.float() / 255.0

        frames = F.interpolate(frames, size=(224, 224), mode='bilinear', align_corners=False)

        return frames, label
from torch.utils.data import DataLoader

dataset = VideoDatasetV3("/content/dataset")
loader = DataLoader(dataset, batch_size=2, shuffle=True)
# =========================
# IMPORTS
# =========================
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split # ✅ Added random_split
from decord import VideoReader, cpu
import os
from tqdm import tqdm
from transformers import VideoMAEForVideoClassification

# =========================
# DATASET (FIXED)
# =========================
class VideoDataset(Dataset):
    def __init__(self, root_dir, num_frames=16):
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.samples = []
        self.labels = []

        classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(classes)}

        for cls in classes:
            cls_path = os.path.join(root_dir, cls)
            for file in os.listdir(cls_path):
                if file.endswith((".mp4", ".avi", ".mov")):
                    self.samples.append(os.path.join(cls_path, file))
                    self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path = self.samples[idx]
        label = self.labels[idx]

        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)

        indices = torch.linspace(0, total_frames - 1, self.num_frames).long()
        frames = vr.get_batch(indices).asnumpy()

        # (T, H, W, C) → (T, C, H, W)
        frames = torch.tensor(frames).permute(0, 3, 1, 2)
        frames = frames.float() / 255.0

        # Resize to 224x224
        frames = F.interpolate(frames, size=(224, 224), mode='bilinear', align_corners=False)

        return frames, label


# =========================
# DATA LOADER
# =========================
full_dataset = VideoDataset("/content/dataset", num_frames=16)

# ✅ Split dataset into training and validation sets
train_size = int(0.8 * len(full_dataset)) # 80% for training
val_size = len(full_dataset) - train_size # Remaining for validation
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)  # ✅ Train DataLoader
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)    # ✅ Validation DataLoader

print(f"Total videos: {len(full_dataset)}")
print(f"Training videos: {len(train_dataset)}")
print(f"Validation videos: {len(val_dataset)}")


# =========================
# MODEL
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VideoMAEForVideoClassification.from_pretrained(
    "MCG-NJU/videomae-base-finetuned-kinetics",
    num_labels=2,
    ignore_mismatched_sizes=True
)

model.to(device)


# =========================
# 🔥 SPEED BOOST (FREEZE BACKBONE)
# =========================
for param in model.videomae.parameters():
    param.requires_grad = False


# =========================
# TRAINING SETUP
# =========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=5e-5)

EPOCHS = 5


# =========================
# TRAIN LOOP
# =========================
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    loop = tqdm(train_loader) # ✅ Use train_loader

    for videos, labels in loop:
        videos = videos.to(device)
        labels = labels.to(device)

        outputs = model(videos).logits
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        loop.set_description(f"Epoch {epoch+1}")
        loop.set_postfix(loss=loss.item())

    print(f"Epoch {epoch+1} Training Loss: {total_loss/len(train_loader)}") # ✅ Updated print statement


