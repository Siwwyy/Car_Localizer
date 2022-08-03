import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.metrics import confusion_matrix
sns.set_style('darkgrid')
from model_vehicle_classifier import VehicleClassifier
from custom_functions import get_mean_and_std, binary_acc, get_mean_std
# from Model import Model_First
np.random.seed(0)
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("In our project we're using =>", device)
root_dir = "/Car_Localizer/Car_Localizer/Training"
print("The data are stored here =>", root_dir)

#Initialize raw training dataset:
train_dataset_raw = datasets.ImageFolder(root = root_dir + "/train",
        transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()]))

#data_raw is used to calculate the mean and std metrics:
train_loader_raw = DataLoader(dataset=train_dataset_raw, shuffle=False, batch_size=len(train_dataset_raw))
data_raw = next(iter(train_loader_raw))

#Initialize transformations as a dict:
image_transforms = {
    "train": transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(
        mean=data_raw[0].mean(),
        std=data_raw[0].std())
    ]),
    "test": transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()])
}

#Initialize Datasets:
train_dataset = datasets.ImageFolder(root = root_dir + "/train", transform = image_transforms["train"])
test_dataset = datasets.ImageFolder(root = root_dir + "/test", transform = image_transforms["test"])

#idx2class is going to be used in confusion matrix:
idx2class = {v: k for k, v in train_dataset.class_to_idx.items()}

#Get Train and Validation Samples:
train_dataset_indices = list(range(len(train_dataset)))
np.random.shuffle(train_dataset_indices)
val_split_index = int(np.floor(0.2 * len(train_dataset)))
train_idx, val_idx = train_dataset_indices[val_split_index:], train_dataset_indices[:val_split_index]
train_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(val_idx)

#Train, Validation, and Test Dataloader
train_loader = DataLoader(dataset=train_dataset, shuffle=False, batch_size=8, sampler=train_sampler)
val_loader = DataLoader(dataset=train_dataset, shuffle=False, batch_size=1, sampler=val_sampler)
test_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=1)

#Initialize the model
model = VehicleClassifier()
model.to(device)
print(model)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.008)

accuracy_stats = {'train': [], "val": []}
loss_stats = {'train': [], "val": []}

#Training loop - tqdm was used in order to check the progress meters
print("Training in progres ...")
for e in tqdm(range(1, 21)):
    # TRAINING
    train_epoch_loss = 0
    train_epoch_acc = 0
    model.train()
    for X_train_batch, y_train_batch in train_loader:
        X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
        optimizer.zero_grad()
        y_train_pred = model(X_train_batch).squeeze()
        train_loss = criterion(y_train_pred, y_train_batch)
        train_acc = binary_acc(y_train_pred, y_train_batch)
        train_loss.backward()
        optimizer.step()
        train_epoch_loss += train_loss.item()
        train_epoch_acc += train_acc.item()
    # VALIDATION
    with torch.no_grad():
        model.eval()
        val_epoch_loss = 0
        val_epoch_acc = 0
        for X_val_batch, y_val_batch in val_loader:
            X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
            y_val_pred = model(X_val_batch).squeeze()
            y_val_pred = torch.unsqueeze(y_val_pred, 0)
            val_loss = criterion(y_val_pred, y_val_batch)
            val_acc = binary_acc(y_val_pred, y_val_batch)
            val_epoch_loss += val_loss.item()
            val_epoch_acc += val_acc.item()
    loss_stats['train'].append(train_epoch_loss/len(train_loader))
    loss_stats['val'].append(val_epoch_loss/len(val_loader))
    accuracy_stats['train'].append(train_epoch_acc/len(train_loader))
    accuracy_stats['val'].append(val_epoch_acc/len(val_loader))
    #It displays the progress of calculation's results:
    print(f'Epoch {e+0:02}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Val Loss: {val_epoch_loss/len(val_loader):.5f} | Train Acc: {train_epoch_acc/len(train_loader):.3f}| Val Acc: {val_epoch_acc/len(val_loader):.3f}')


#Visualize Loss and Accuracy
train_val_acc_df = pd.DataFrame.from_dict(accuracy_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
train_val_loss_df = pd.DataFrame.from_dict(loss_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(30,10))
sns.lineplot(data=train_val_acc_df, x = "epochs", y="value", hue="variable",  ax=axes[0]).set_title('Train-Val Accuracy/Epoch')
sns.lineplot(data=train_val_loss_df, x = "epochs", y="value", hue="variable", ax=axes[1]).set_title('Train-Val Loss/Epoch')


#Test - we need to test how our model fared. At the end I performed the confusion matrix.
y_pred_list = []
y_true_list = []
with torch.no_grad():
    for x_batch, y_batch in tqdm(test_loader):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        y_test_pred = model(x_batch)
        _, y_pred_tag = torch.max(y_test_pred, dim = 1)
        y_pred_list.append(y_pred_tag.cpu().numpy())
        y_true_list.append(y_batch.cpu().numpy())

# Initialize visualization:
y_pred_list = [i[0][0][0] for i in y_pred_list]
y_true_list = [i[0] for i in y_true_list]
confusion_matrix_df = pd.DataFrame(confusion_matrix(y_true_list, y_pred_list)).rename(columns=idx2class, index=idx2class)
fig, ax = plt.subplots(figsize=(7,5))
sns.heatmap(confusion_matrix_df, annot=True, ax=ax)