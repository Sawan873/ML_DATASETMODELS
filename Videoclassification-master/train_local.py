import os
import sys
import copy
import cv2
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import zipfile
from torch.autograd import Variable
import gdown

# 1. LOCAL DIRECTORIES
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ZIP_PATH = os.path.join(BASE_DIR, "data.zip")
DATA_DIR = os.path.join(BASE_DIR, "crime")

print("--- Starting Local Training Initialization ---")
# 2. DOWNLOAD DATASET LOCALLY IF REQUIRED
if not os.path.exists(DATA_DIR):
    print("Downloading UCF Crime Subset from secure cloud storage...")
    gdown.download(id='1TdfOz-MVrA_XP8h8oXd8IBTvOXwbFN1Y', output=ZIP_PATH, quiet=False)
    print("Extracting dataset...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(BASE_DIR)
else:
    print("Dataset already downloaded locally.")

# 3. LABEL ENCODING
classes = os.listdir(DATA_DIR)
# Filter out any non-directories or hidden files
classes = [c for c in classes if os.path.isdir(os.path.join(DATA_DIR, c))]
classes.sort()

decoder = {cls: i for i, cls in enumerate(classes)}
encoder = {i: cls for i, cls in enumerate(classes)}

id = []
for idx_i, c_name in enumerate(classes):
    p1 = os.path.join(DATA_DIR, c_name)
    for j in os.listdir(p1):
        p2 = os.path.join(p1, j)
        id.append((c_name, p2))

# 4. DATASET DEFINITION
class video_dataset(Dataset):
    def __init__(self, frame_list, sequence_length=16, transform=None):
        self.frame_list = frame_list
        self.transform = transform
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.frame_list)

    def __getitem__(self, idx):
        label, path = self.frame_list[idx]
        img = cv2.imread(path)
        seq_img = []
        for i in range(16):
            img1 = img[:, 128*i : 128*(i+1), :]
            if self.transform:
                img1 = self.transform(img1)
            seq_img.append(img1)
        seq_image = torch.stack(seq_img)
        seq_image = seq_image.reshape(3, 16, im_size, im_size)
        return seq_image, decoder[label]

im_size = 128
mean = [0.4889, 0.4887, 0.4891]
std = [0.2074, 0.2074, 0.2074]

train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((im_size, im_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Use smaller batch_size for Local CPU memory constraints
train_data = video_dataset(id, sequence_length=16, transform=train_transforms)
train_loader = DataLoader(train_data, batch_size=4, num_workers=0, shuffle=True)
dataloaders = {'train': train_loader}

# 5. ENVIRONMENT / HARDWARE SETUP
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Hardware assigned for training: {device.upper()}")

from model import resnet50
from clr import *

model = resnet50(class_num=8).to(device)
cls_criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)

num_epochs = 10
onecyc = OneCycle(len(train_loader) * num_epochs, 1e-3)

# 6. TRAINING LOOP
iteration = 0
for epoch in range(num_epochs):
    print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
    for phase in dataloaders.keys():
        epoch_metrics = {"loss": [], "acc": []}
        for batch_i, (X, y) in enumerate(dataloaders[phase]):
            image_sequences = Variable(X.to(device), requires_grad=True)
            labels = Variable(y.to(device), requires_grad=False)
            
            optimizer.zero_grad()
            predictions = model(image_sequences)
            loss = cls_criterion(predictions, labels)
            acc = 100 * (predictions.detach().argmax(1) == labels).cpu().numpy().mean()
            
            loss.backward()
            optimizer.step()
            
            epoch_metrics["loss"].append(loss.item())
            epoch_metrics["acc"].append(acc)
            
            if phase == 'train':
                lr, mom = onecyc.calc()
                update_lr(optimizer, lr)
                update_mom(optimizer, mom)
                
            sys.stdout.write(
                f"\r[Epoch {epoch+1}/{num_epochs}] [Batch {batch_i}/{len(dataloaders[phase])}] "
                f"[Loss: {loss.item():.4f} ({np.mean(epoch_metrics['loss']):.4f}), "
                f"Acc: {acc:.2f}% ({np.mean(epoch_metrics['acc']):.2f}%)]   "
            )
            sys.stdout.flush()

        print(f"\n{phase.capitalize()} Accuracy: {np.mean(epoch_metrics['acc']):.2f}%")
        
        # Save dynamically to the Master folder natively
        save_path = os.path.join(BASE_DIR, f'c3d_{epoch}.h5')
        torch.save(model.state_dict(), save_path)
        print(f"Checkpoint saved: {save_path}")

print("\nTraining completed successfully! Local models generated.")
