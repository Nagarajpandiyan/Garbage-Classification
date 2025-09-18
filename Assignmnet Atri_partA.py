#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Standard Imports 
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#PyTorch Imports 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# WandB 
import wandb
from pandas.plotting import parallel_coordinates

# WandB Login
wandb.login(key='0530bbf36999cf23c9faffc230ca42a929ee045b')
entity = 'cs24s023-iitm-ac-in'
project = 'garbage_classification'
os.environ["WANDB_NOTEBOOK_NAME"] = "garbage_cnn.py"


# In[2]:


# ataset & Transformations 
DATA_PATH = '/home/nagaraj/Garbage_classification_files/Garbage classification'
CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Data augmentation for training
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(128, scale=(0.8,1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# No augmentation for validation/test
test_transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# Load dataset
dataset = datasets.ImageFolder(DATA_PATH, transform=train_transform)
train_len = int(0.8 * len(dataset))
val_test_len = len(dataset) - train_len
train_data, val_test_data = random_split(dataset, [train_len, val_test_len])
val_len = test_len = val_test_len // 2
val_data, test_data = random_split(val_test_data, [val_len, test_len])

# Replace transforms for val/test
val_data.dataset.transform = test_transform
test_data.dataset.transform = test_transform

# DataLoaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)


# In[3]:


class GarbageCNN(nn.Module):
    """
    Small CNN for Garbage Classification
    - 5 convolutional layers with optional batchnorm and dropout
    - Dense layer followed by output layer (6 classes)
    """
    def __init__(self, filters=[32,64,128,128,256], kernel_size=3,
                 activation='relu', dense_units=512, dropout=0.3, use_bn=True):
        super().__init__()
        self.activation = activation
        self.use_bn = use_bn
        self.layers = nn.ModuleList()
        
        # Convolutional blocks
        in_channels = 3
        for out_channels in filters:
            self.layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=1))
            if use_bn:
                self.layers.append(nn.BatchNorm2d(out_channels))
            self.layers.append(nn.MaxPool2d(2))
            in_channels = out_channels
        
        # Flatten
        self.flatten = nn.Flatten()
        
        # Compute flattened size
        dummy_input = torch.zeros(1,3,128,128)
        for layer in self.layers:
            dummy_input = layer(dummy_input)
        fc_input_features = dummy_input.numel()
        
        # Dense layers
        self.fc1 = nn.Linear(fc_input_features, dense_units)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dense_units, len(CLASSES))
    
    def forward(self, x):
        # Pass through conv layers
        for layer in self.layers:
            x = layer(x)
            if isinstance(layer, (nn.Conv2d, nn.BatchNorm2d)):
                x = self._apply_activation(x)
        
        # Flatten and dense layers
        x = self.flatten(x)
        x = self._apply_activation(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    def _apply_activation(self, x):
        if self.activation=='relu': return F.relu(x)
        elif self.activation=='gelu': return F.gelu(x)
        elif self.activation=='silu': return F.silu(x)
        elif self.activation=='mish': return x * torch.tanh(F.softplus(x))
        return x


# In[4]:


# Parameter & FLOPs 
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def estimate_flops(model, input_size=(1,3,128,128), device='cpu'):
    flops = 0
    x = torch.zeros(input_size).to(device)
    for layer in model.layers:
        if isinstance(layer, nn.Conv2d):
            out_ch, in_ch, k, _ = layer.weight.shape
            h, w = x.shape[2], x.shape[3]
            flops += 2 * h * w * in_ch * out_ch * k * k
        x = layer(x)
    in_f, out_f = model.fc1.in_features, model.fc1.out_features
    flops += 2*in_f*out_f + 2*out_f*model.fc2.out_features
    return flops


# In[5]:


class GuidedBackprop:
    """
    Generates gradients for visualization of neuron activations.
    """
    def __init__(self, model):
        self.model = model.eval()
        self._register_hooks()
    
    def _register_hooks(self):
        def hook_fn(module, grad_in, grad_out):
            return (torch.clamp(grad_in[0], min=0.0),)
        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                module.register_backward_hook(hook_fn)
    
    def generate_gradients(self, input_img, target_neuron):
        input_img = input_img.unsqueeze(0).requires_grad_(True)
        self.model.zero_grad()
        output = self.model(input_img)
        output[0, target_neuron].backward()
        return input_img.grad.data[0]


# In[6]:


def train_sweep():
    """
    Training function for WandB sweep.
    Includes: training, validation, test evaluation,
    first-layer filter visualization, and guided backprop.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wandb.init()
    cfg = wandb.config

    #Model Setup
    model = GarbageCNN(
        filters=cfg.filters,
        activation=cfg.activation,
        dense_units=cfg.dense_neurons,
        dropout=cfg.dropout,
        use_bn=cfg.batch_norm
    ).to(device)

    # ----------------- Optimizer & Loss -----------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    criterion = nn.CrossEntropyLoss()

    # Log model stats
    wandb.log({
        'trainable_params': count_parameters(model),
        'estimated_flops': estimate_flops(model, device=device)
    })

    #Training Loop 
    for epoch in range(30):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        train_acc = correct / total

        # Validation
        model.eval()
        correct_val, total_val = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, preds = outputs.max(1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)
        val_acc = correct_val / total_val

        wandb.log({
            'epoch': epoch+1,
            'train_loss': total_loss/len(train_loader),
            'train_acc': train_acc,
            'val_acc': val_acc
        })
        scheduler.step(val_acc)

    # Test Accuracy & Visualization 
    model.eval()
    correct_test, total_test = 0, 0
    all_imgs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = outputs.max(1)
            correct_test += (preds == labels).sum().item()
            total_test += labels.size(0)
            all_imgs.append(imgs.cpu())
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    test_acc = correct_test / total_test
    print(f'Test Accuracy: {test_acc*100:.2f}%')
    wandb.log({'test_accuracy': test_acc})

    # 10x3 Test Image Grid
    all_imgs = torch.cat(all_imgs)
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    fig, axes = plt.subplots(10, 3, figsize=(10,30))
    axes = axes.flatten()
    for i in range(min(30, len(all_imgs))):
        img = all_imgs[i].permute(1,2,0).numpy()
        img = np.clip(img * np.array([0.229,0.224,0.225]) + np.array([0.485,0.456,0.406]), 0, 1)
        axes[i].imshow(img)
        axes[i].set_title(f"Pred: {CLASSES[all_preds[i]]}\nTrue: {CLASSES[all_labels[i]]}")
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()
    wandb.log({"test_samples": wandb.Image(fig)})


# In[7]:


def visualize_filters_and_guided_backprop(model, test_data):
    """
    Visualize first convolutional layer filters and guided backprop
    on 10 neurons of the last conv layer (CONV5).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    # ----------------- First-layer filters -----------------
    first_conv = next((l for l in model.layers if isinstance(l, nn.Conv2d)), None)
    if first_conv:
        weights = first_conv.weight.data.cpu()
        num_filters = weights.shape[0]
        fig, axes = plt.subplots(4, 8, figsize=(12,6))
        axes = axes.flatten()
        for i in range(min(num_filters, len(axes))):
            filt = weights[i].permute(1,2,0)
            filt = (filt - filt.min()) / (filt.max() - filt.min())
            axes[i].imshow(filt)
            axes[i].axis('off')
        plt.suptitle("First Layer Filters")
        plt.show()
        wandb.log({"first_layer_filters": wandb.Image(fig)})
    
    #Guided Backprop (CONV5 neurons)
    conv_layers = [l for l in model.layers if isinstance(l, nn.Conv2d)]
    conv5 = conv_layers[-1]
    gb = GuidedBackprop(model)

    sample_img, _ = test_data[0]
    sample_img = sample_img.to(device)
    
    fig, axes = plt.subplots(2, 5, figsize=(15,6))
    axes = axes.flatten()
    for i in range(min(10, conv5.out_channels)):
        grads = gb.generate_gradients(sample_img, target_neuron=i)
        grads = grads.permute(1,2,0).cpu().numpy()
        grads = (grads - grads.min()) / (grads.max() - grads.min())
        axes[i].imshow(grads)
        axes[i].axis('off')
        axes[i].set_title(f"Neuron {i}")
    plt.suptitle("Guided Backprop - CONV5 Neurons")
    plt.show()
    wandb.log({"guided_backprop_conv5": wandb.Image(fig)})


# In[8]:


#Sweep Configuration
sweep_cfg = {
    'method': 'random',
    'metric': {'name': 'val_acc', 'goal': 'maximize'},
    'parameters': {
        'filters': {'values': [[32,64,128,128,256],[64,128,256,256,512]]},
        'activation': {'values': ['relu','gelu','silu','mish']},
        'dropout': {'values': [0.2,0.3]},
        'batch_norm': {'values': [True,False]},
        'dense_neurons': {'values': [256,512,1024]},
        'lr': {'values': [0.001,0.0005]}
    }
}

sweep_id = wandb.sweep(sweep_cfg, project=project, entity=entity)
print(f"Sweep created: {sweep_id}")
wandb.agent(sweep_id, train_sweep, count=20)

# Sweep Analysis 
api = wandb.Api()
sweep = api.sweep(f"{entity}/{project}/sweeps/{sweep_id}")
runs = sweep.runs
data = []

for run in runs:
    cfg = run.config
    summary = run.summary
    data.append({
        'filters': str(cfg['filters']),
        'activation': cfg['activation'],
        'dropout': cfg['dropout'],
        'batch_norm': cfg['batch_norm'],
        'dense_neurons': cfg['dense_neurons'],
        'lr': cfg['lr'],
        'val_acc': summary.get('val_acc', np.nan),
        'test_accuracy': summary.get('test_accuracy', np.nan)
    })

df = pd.DataFrame(data)
print(df)

# Parallel Coordinates Plot
plt.figure(figsize=(12,6))
parallel_coordinates(df, 'activation', cols=['val_acc','dropout','dense_neurons','lr'])
plt.title("Parallel Coordinates Plot of Sweep")
plt.show()

# Correlation Table
numeric_cols = ['dropout','dense_neurons','lr','val_acc','test_accuracy']
corr_table = df[numeric_cols].corr()
print("\nCorrelation Table:\n", corr_table)


# In[ ]:




