#cd C:\Users\marcb\OneDrive\Desktop\Kaggle_Competitions\Kaggle-Competitions\HMS-harmful-brain-activity
import gc
import os
import random
import warnings
import numpy as np
import pandas as pd
from IPython.display import display

# PyTorch for deep learning
import timm
import torch
import torch.nn as nn  
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

# torchvision for image processing and augmentation
import torchvision.transforms as transforms

# Suppressing minor warnings to keep the output clean
warnings.filterwarnings('ignore', category=Warning)

# Reclaim memory no longer in use.
gc.collect()


# Configuration class containing hyperparameters and settings
class Config:
    seed = 42 
    image_transform = transforms.Resize((440,440))  
    batch_size = 50
    num_epochs = 10
    num_folds = 5

# Set the seed for reproducibility across multiple libraries
def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
set_seed(Config.seed)

# Define the 'Kullback Leibler Divergence' loss function
def KL_loss(p,q):
    epsilon=10**(-15)
    p=torch.clip(p,epsilon,1-epsilon)
    q = nn.functional.log_softmax(q,dim=1)
    return torch.mean(torch.sum(p*(torch.log(p)-q),dim=1))

# Reclaim memory no longer in use.
gc.collect()

# Load training data

PATH = "C:\\Users\\marcb\\OneDrive\\Desktop\\Kaggle_Competitions\\Kaggle-Competitions\\HMS-harmful-brain-activity\\"
test_eeg = PATH+'hms-harmful-brain-activity-classification/train_eegs/'
test_csv = PATH+'hms-harmful-brain-activity-classification/train.csv'

train_df = pd.read_csv(test_csv)

# Define labels for classification
labels = ['seizure', 'lpd', 'gpd', 'lrda', 'grda', 'other']

# Initialize an empty DataFrame for storing features
train_feats = pd.DataFrame()

# Aggregate votes for each label and merge into train_feats DataFrame
for label in labels:
    # Group by 'spectrogram_id' and sum the votes for the current label
    group = train_df[f'{label}_vote'].groupby(train_df['spectrogram_id']).sum()

    # Create a DataFrame from the grouped data
    label_vote_sum = pd.DataFrame({'spectrogram_id': group.index, f'{label}_vote_sum': group.values})

    # Initialize train_feats with the first label or merge subsequent labels
    if label == 'seizure':
        train_feats = label_vote_sum
    else:
        train_feats = train_feats.merge(label_vote_sum, on='spectrogram_id', how='left')

# Add a column to sum all votes
train_feats['total_vote'] = 0
for label in labels:
    train_feats['total_vote'] += train_feats[f'{label}_vote_sum']

# Calculate and store the normalized vote for each label
for label in labels:
    train_feats[f'{label}_vote'] = train_feats[f'{label}_vote_sum'] / train_feats['total_vote']

# Select relevant columns for the training features
choose_cols = ['spectrogram_id']
for label in labels:
    choose_cols += [f'{label}_vote']
train_feats = train_feats[choose_cols]

# Add a column with the path to the spectrogram files
train_feats['path'] = train_feats['spectrogram_id'].apply(lambda x: PATH+'hms-harmful-brain-activity-classification/train_spectrograms/' + str(x) + ".parquet")

# Reclaim memory no longer in use.
gc.collect()

from joblib import Parallel, delayed


def get_data(path, batch_size=Config.batch_size):
    # Set a small epsilon to avoid division by zero
    # eps = 1e-8

    # # Initialize a list to store batch data
    # #batch_data = []

    # # Iterate over each path in the provided paths
    # #for path in paths:
        # # Read data from parquet file
    # data = pd.read_parquet(path)
    # #dat_mean = data.mean()
    # # Fill missing values, remove time column, and transpose
    # data = data.values[:, 1:].T

    # # Clip values and apply logarithmic transformation
    # data = np.clip(data, np.exp(-6), np.exp(10))
    # data = np.log(data)

    # # Normalize the data
    # data_max = np.nanmax(data)
    # data_min = np.nanmin(data)
    # data = (data - data_min) / (data_max - data_min + eps)
    

    # # Convert data to a PyTorch tensor and apply transformations
    # data_tensor = torch.unsqueeze(torch.Tensor(data), dim=0)
    # #data_tensor = data_tensor.repeat(3,1,1)
    # data_tensor = torch.nan_to_num(data_tensor)
    # data = Config.image_transform(data_tensor)
    eps = 1e-8
    # Read and preprocess spectrogram data
    data = pd.read_parquet(path)
    data = data.fillna(-1).values[:, 1:].T
    data = np.clip(data, np.exp(-6), np.exp(10))
    data = np.log(data)
    data_max = data.max(axis=(0, 1))
    data_min = data.min(axis=(0, 1))
    data = (data - data_min) / (data_max - data_min + eps)
    
    # Normalize the data
    # data_mean = data.mean(axis=(0, 1))
    # data_std = data.std(axis=(0, 1))
    # data = (data - data_mean) / (data_std + eps)
    data_tensor = torch.unsqueeze(torch.Tensor(data), dim=0)
    data = Config.image_transform(data_tensor)

    # Append the processed data to the batch_data list
    #batch_data.append(data)

    # Stack all the batch data into a single tensor
    #batch_data = torch.stack(batch_data)

    # Return the batch data
    return data
parallel = Parallel(n_jobs=14)
data = parallel(delayed(get_data)(path) for path in train_feats.path)

from joblib.externals.loky import get_reusable_executor
get_reusable_executor().shutdown(wait=True)
data

from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
def img_transform(img):
    transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            #A.CoarseDropout(max_holes=4,max_height=54,max_width=18,fill_value=0,p=0.5),
            A.RandomBrightnessContrast(p=0.5, brightness_limit=0.1, contrast_limit=0.1),
            A.Blur(blur_limit=3,p=.5),
            #A.transforms.CLAHE(p=.5),
            #A.ChannelDropout((1,1), p=.5), 
            #A.transforms.ChannelShuffle(),
            A.transforms.PixelDropout(dropout_prob=0.005,p=.5), 
            #A.transforms.Sharpen() 
        ])
    transforms(image=img)
    return img

class CustomDataset_aug(Dataset):
    def __init__(self, train_feats, data):
        self.train_feats = train_feats
        self.data = data

    def __len__(self):
        return len(self.train_feats)

    def __getitem__(self, index):        
        x = self.data[index]
        x = img_transform(np.array(x))
        #x = x.permute(1,0)
        y = self.train_feats[['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']].iloc[index].values
        return torch.tensor(x), y

class CustomDataset(Dataset):
    def __init__(self, train_feats, data):
        self.train_feats = train_feats
        self.data = data
    def __len__(self):
        return len(self.train_feats)
    def __getitem__(self, index):              
        x = self.data[index]
        y = self.train_feats[['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']].iloc[index].values      
        return x, y

# dataset = CustomDataset( train_feats, data)
# train_dataloader = DataLoader(dataset, batch_size=6, shuffle=True,  num_workers=0, drop_last=True)
# x,y = next(iter(train_dataloader))

def redo_classifier(model):
    num_in_features = model.get_classifier().in_features
    hidden_size = 64
    n_classes=6
    dropout_rate=.2
    d2 = .1
    for name, param in model.named_parameters():
        ijk=0
        #print (name)
        
    # Replace the existing classifier. It's named: classifier
    if "head.fc" in name:
        model.head.fc = nn.Sequential(
        nn.Linear(in_features=num_in_features, out_features=hidden_size, bias=True),
        nn.SiLU(),
        #nn.ReLU(inplace=True),
        #nn.RReLU(lower=0.05, upper=0.3333333333333333, inplace=True),
        #nn.GELU(),
        # nn.BatchNorm1d(hidden_size),
        nn.Dropout(d2),
        nn.Linear(hidden_size, 32, bias=True),
        #nn.Softmax(dim=0),
        # nn.BatchNorm1d(32),
        nn.Linear(32, out_features=n_classes, bias=True),
        #nn.Softmax(dim=0)
        )
    elif "fc" in name:
        model.fc = nn.Sequential(
        nn.Linear(in_features=num_in_features, out_features=hidden_size, bias=True),
        nn.Hardswish(),
        #nn.ReLU(inplace=True),
        #nn.RReLU(lower=0.05, upper=0.3333333333333333, inplace=True),
        #nn.GELU(),
        # nn.BatchNorm1d(hidden_size),
        nn.Dropout(d2),
        #nn.Linear(hidden_size, 32, bias=True),
        #nn.SiLU(),
        # nn.BatchNorm1d(32),
        nn.Linear(hidden_size, out_features=n_classes, bias=True),
        )
    elif "classifier" in name:
        model.classifier = nn.Sequential(
        nn.Linear(in_features=num_in_features, out_features=hidden_size, bias=True),
        
        #nn.ReLU(inplace=True),
        #nn.RReLU(lower=0.05, upper=0.3333333333333333, inplace=True),
        #nn.GELU(),
        # nn.BatchNorm1d(hidden_size),
        nn.Dropout(d2),
        nn.SiLU(),
        #nn.Linear(hidden_size, 32, bias=True),
        #nn.Hardswish(),
        # nn.BatchNorm1d(32),
        nn.Linear(hidden_size, out_features=n_classes, bias=True),
        )
    elif "head" in name:
        model.head = nn.Sequential(
        #nn.Dropout(dropout_rate, inplace=True),
        nn.Linear(in_features=num_in_features, out_features=hidden_size, bias=True),
        nn.Hardswish(),
        #nn.ReLU(inplace=True),
        #nn.RReLU(lower=0.05, upper=0.3333333333333333, inplace=True),
        #nn.GELU(),
        # nn.BatchNorm1d(hidden_size),
        nn.Dropout(d2),
        #nn.Linear(hidden_size, 32, bias=True),
        #nn.Hardswish(),
        # nn.BatchNorm1d(32),
        nn.Linear(hidden_size, out_features=n_classes, bias=True),
        )
    #elif "neck" in name:
    #    model.head = nn.Sequential(
    #    nn.AdaptiveAvgPool2d((64,1)),
    #    nn.Dropout(dropout_rate),
    #    nn.Linear(in_features=1, out_features=64, bias=False),
        #nn.LeakyReLU(.1,inplace=True),
        #nn.Dropout(dropout_rate),
        #nn.Linear(in_features=hidden_size, out_features=1, bias=False),
        #nn.Sigmoid())
    print (name)
    print (model)
    return model

# model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=6, in_chans=1)
# model = redo_classifier(model)
#model

# Determine device availability
import tqdm
from sklearn.model_selection import train_test_split

import albumentations as A
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Assuming train_feats is defined and contains the training features and labels
total_idx = np.arange(len(train_feats))
np.random.shuffle(total_idx)

gc.collect()
criterion = nn.KLDivLoss(reduction='batchmean')
# Cross-validation loop
for fold in range(3):
    # Split data into train and test sets for this fold
    train_idx, test_idx = train_test_split(total_idx, test_size=0.1)
    # test_idx = total_idx[fold * len(total_idx) // (2*Config.num_folds):(fold + 1) * len(total_idx) // (2*Config.num_folds)]
    # train_idx = np.array([idx for idx in total_idx if idx not in test_idx])
    print('train length: ', len(train_idx), ' test length: ', len(test_idx))
    train_data = [data[i] for i in train_idx]
    test_data = [data[i] for i in test_idx]
    # keys = list(data.item().keys())
    # test_keys = [keys[i] for i in test_idx]
    # train_keys = [keys[i] for i in train_idx]
    train_dataset = CustomDataset_aug(train_feats.iloc[train_idx], train_data)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True,  num_workers=0, drop_last=False)
    test_dataset = CustomDataset(train_feats.iloc[test_idx], test_data)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False,  num_workers=0, drop_last=False)
    torch.cuda.empty_cache()

    # Initialize EfficientNet-B0 model with pretrained weights
    #model = timm.create_model('convnext_base', pretrained=True,drop_rate=.2,drop_path_rate=.2 ,num_classes=1000, in_chans=1)
    model = timm.create_model('tf_efficientnet_b3.ns_jft_in1k', in_chans=1, drop_path_rate=.2,num_classes=1000,pretrained=True)
    #model = model.eval()

    # get model specific transforms (normalization, resize)
    # data_config = timm.data.resolve_model_data_config(model)
    # transforms = timm.data.create_transform(**data_config, is_training=False)
    model = redo_classifier(model)
    model.to(device)
    

    optimizer = optim.NAdam(model.parameters(), lr=0.0002, weight_decay=0.02)
    scheduler = CosineAnnealingLR(optimizer, T_max=12)

    best_test_loss = float('inf')
    train_losses = []
    test_losses = []

    print(f"Starting training for fold {fold + 1}")

    # Training loop
    for epoch in range(12):
        model.train()
        train_loss = []
        # random_num = np.arange(len(train_idx))
        # np.random.shuffle(random_num)
        # train_idx = train_idx[random_num]

        # Iterate over batches in the training set
        count=0
        for x,y in train_dataloader:
            count=count+1
            optimizer.zero_grad()
            #x = transforms(x)
            #x = x.unsqueeze(1)

            train_pred = model(x.to(device))
            loss = KL_loss(y.to(device),train_pred)
            #loss = torch.abs(criterion(train_pred,y.to(device)))
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            if count%50==0:
                print(loss.item())

        epoch_train_loss = np.mean(train_loss)
        train_losses.append(epoch_train_loss)
        print(f"Epoch {epoch + 1}: Train Loss = {epoch_train_loss:.2f}")

        scheduler.step()

        # Evaluation loop
        model.eval()
        test_loss = []
        with torch.no_grad():
            for x,y in test_dataloader:
                #x = x.unsqueeze(1)
                #x = transforms(x)

                test_pred = model(x.to(device))
                loss = KL_loss(y.to(device),test_pred)
                #loss = torch.abs(criterion(test_pred,y.to(device)))
                test_loss.append(loss.item())

        epoch_test_loss = np.mean(test_loss)
        test_losses.append(epoch_test_loss)
        print(f"Epoch {epoch + 1}: Test Loss = {epoch_test_loss:.2f}")

        # Save the model if it has the best test loss so far
        if epoch_test_loss < best_test_loss:
            best_test_loss = epoch_test_loss
            torch.save(model.state_dict(), f"efficientnet_b0_fold{fold}.pth")

        gc.collect()

    print(f"Fold {fold + 1} Best Test Loss: {best_test_loss:.2f}")
	
	