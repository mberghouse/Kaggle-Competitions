#cd C:\Users\marcb\OneDrive\Desktop\Kaggle_Competitions\Kaggle-Competitions\HMS-harmful-brain-activity

import gc
import os
import random
import warnings
import numpy as np
import pandas as pd
from IPython.display import display
from joblib.externals.loky import get_reusable_executor
get_reusable_executor().shutdown(wait=True)
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

PATH_OR = "C:\\Users\\marcb\\OneDrive\\Desktop\\Kaggle_Competitions\\Kaggle-Competitions\\HMS-harmful-brain-activity\\"
test_eeg = PATH_OR+'hms-harmful-brain-activity-classification/train_eegs/'
test_specs = PATH_OR+'hms-harmful-brain-activity-classification/train_spectrograms/'
test_csv = PATH_OR+'hms-harmful-brain-activity-classification/train.csv'

df = pd.read_csv(test_csv)
TARGETS = df.columns[-6:]
print('Train shape:', df.shape )
print('Targets', list(TARGETS))
df.head()

train = df.groupby('eeg_id')[['spectrogram_id','spectrogram_label_offset_seconds']].agg(
    {'spectrogram_id':'first','spectrogram_label_offset_seconds':'min'})
train.columns = ['spec_id','min']

tmp = df.groupby('eeg_id')[['spectrogram_id','spectrogram_label_offset_seconds']].agg(
    {'spectrogram_label_offset_seconds':'max'})
train['max'] = tmp

tmp = df.groupby('eeg_id')[['patient_id']].agg('first')
train['patient_id'] = tmp

tmp = df.groupby('eeg_id')[TARGETS].agg('sum')
for t in TARGETS:
    train[t] = tmp[t].values
    
y_data = train[TARGETS].values
y_data = y_data / y_data.sum(axis=1,keepdims=True)
train[TARGETS] = y_data

tmp = df.groupby('eeg_id')[['expert_consensus']].agg('first')
train['target'] = tmp

train = train.reset_index()
print('Train non-overlapp eeg_id shape:', train.shape )
train.head()

READ_SPEC_FILES = True

# READ ALL SPECTROGRAMS
PATH = PATH_OR+'hms-harmful-brain-activity-classification/train_spectrograms/'
files = os.listdir(PATH)
print(f'There are {len(files)} spectrogram parquets')

    
import pywt, librosa
import pandas as pd, numpy as np
import matplotlib.pyplot as plt

USE_WAVELET = None 

NAMES = ['LL','LP','RP','RR']

FEATS = [['Fp1','F7','T3','T5','O1'],
         ['Fp1','F3','C3','P3','O1'],
         ['Fp2','F8','T4','T6','O2'],
         ['Fp2','F4','C4','P4','O2']]

# DENOISE FUNCTION
def maddest(d, axis=None):
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)

def denoise(x, wavelet='haar', level=1):    
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1/0.6745) * maddest(coeff[-level])

    uthresh = sigma * np.sqrt(2*np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])

    ret=pywt.waverec(coeff, wavelet, mode='per')
    
    return ret

def spectrogram_from_eeg(parquet_path):
    
    # LOAD MIDDLE 50 SECONDS OF EEG SERIES
    eeg = pd.read_parquet(parquet_path)
    middle = (len(eeg)-10_000)//2
    eeg = eeg.iloc[middle:middle+10_000]
    
    # VARIABLE TO HOLD SPECTROGRAM
    img = np.zeros((128,256,4),dtype='float32')

    signals = []
    for k in range(4):
        COLS = FEATS[k]
        
        for kk in range(4):
        
            # COMPUTE PAIR DIFFERENCES
            x = eeg[COLS[kk]].values - eeg[COLS[kk+1]].values

            # FILL NANS
            m = np.nanmean(x)
            if np.isnan(x).mean()<1: x = np.nan_to_num(x,nan=m)
            else: x[:] = 0

            # DENOISE
            if USE_WAVELET:
                x = denoise(x, wavelet=USE_WAVELET)
            signals.append(x)

            # RAW SPECTROGRAM
            mel_spec = librosa.feature.melspectrogram(y=x, sr=200, hop_length=len(x)//256, 
                  n_fft=1024, n_mels=128, fmin=0, fmax=20, win_length=128)

            # LOG TRANSFORM
            width = (mel_spec.shape[1]//32)*32
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)[:,:width]

            # STANDARDIZE TO -1 TO 1
            mel_spec_db = (mel_spec_db+40)/40 
            img[:,:,k] += mel_spec_db
                
        # AVERAGE THE 4 MONTAGE DIFFERENCES
        img[:,:,k] /= 4.0
        
        
    return img
    
    
from joblib import Parallel, delayed
import pandas as pd
import os

def process_spectrogram(f, PATH2):
    tmp = pd.read_parquet(f'{PATH2}{f}')
    name = int(f.split('.')[0])
    return name, tmp.iloc[:,1:].values

def process_eeg(eeg_id, PATH2):
    img = spectrogram_from_eeg(f'{PATH2}{eeg_id}.parquet')
    return eeg_id, img

PATH2 = PATH_OR+'hms-harmful-brain-activity-classification/train_spectrograms/'
files2 = os.listdir(PATH2)
print(f'There are {len(files2)} test spectrogram parquets')

results = Parallel(n_jobs=16)(delayed(process_spectrogram)(f, PATH2) for f in files2)
spectrograms2 = dict(results)

from joblib.externals.loky import get_reusable_executor
get_reusable_executor().shutdown(wait=True)

PATH2 = PATH_OR+'hms-harmful-brain-activity-classification/train_eegs/'
DISPLAY = 0
EEG_IDS2 = train.eeg_id.unique()

print('Converting Test EEG to Spectrograms...'); print()
results = Parallel(n_jobs=16)(delayed(process_eeg)(eeg_id, PATH2) for  eeg_id in (EEG_IDS2))
all_eegs2 = dict(results)

from joblib.externals.loky import get_reusable_executor
get_reusable_executor().shutdown(wait=True)

import cv2
eegs = np.zeros([17089,128,256,4])
specs = np.zeros([17089,400,400])
for i in range(17089):
    eegs[i] = all_eegs2[train.eeg_id[i]]
del all_eegs2
for i in range(17089):
    specs[i] = cv2.resize(spectrograms2[train.spec_id[i]],(400,400))
del spectrograms2
    
from sklearn.model_selection import train_test_split
eps=1e-8

from torch.utils.data import Dataset, DataLoader
class CustomDataset(Dataset):
    def __init__(self, train_feats, eegs, specs):
        self.train_feats = train_feats
        self.eegs = eegs
        self.specs = specs
    def __len__(self):
        return len(self.train_feats)
    def __getitem__(self, index):              
        x_eeg = self.eegs[index]
        x_eeg=torch.nan_to_num(x_eeg)
        x_eeg = torch.tensor((x_eeg-x_eeg.min())/(x_eeg.max()-x_eeg.min()+eps))
        x_eeg = torch.tensor(x_eeg)
        
        x_spec = self.specs[index]
        x_spec=torch.nan_to_num(x_spec)
        x_spec = torch.tensor((x_spec-x_spec.min())/(x_spec.max()-x_spec.min()+eps))
        x_spec = torch.tensor(x_spec).unsqueeze(0)
        torch.nan_to_num(x_spec)
        y = self.train_feats[['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']].iloc[index].values      
        return x_eeg, x_spec, y
        
        
total_idx = np.arange(len(train))
train_idx, val_idx = train_test_split(total_idx, test_size=0.2)
train_idx

def redo_classifier(model):
    num_in_features = model.get_classifier().in_features
    hidden_size = 64
    n_classes=32
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
    #print (model)
    return model

# model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=6, in_chans=1)
# model = redo_classifier(model)
#model

import torch
import torch.nn as nn

class CombinedModel(nn.Module):
    def __init__(self):
        super(CombinedModel, self).__init__()
        self.model1 = timm.create_model('tf_efficientnet_b1.ns_jft_in1k', in_chans=4, drop_path_rate=.3,num_classes=32,pretrained=True)

        self.model2 = timm.create_model('tf_efficientnet_b1.ns_jft_in1k', in_chans=1, drop_path_rate=.4,num_classes=32,pretrained=True)
    
        self.fc = nn.Linear(64, 6)

    def forward(self, x1, x2):

        # Run the first tensor through the first model
        output1 = F.silu(self.model1(x1.float()))

        # Run the second tensor through the second model
        output2 = F.silu(self.model2(x2.float()))

        # Concatenate the outputs
        combined_output = torch.cat((output1,output2),1)

        # Pass the combined output through the final linear layer
        combined_output = self.fc(combined_output)

        return combined_output

# Determine device availability
import tqdm
from sklearn.model_selection import train_test_split

import albumentations as A
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Assuming train_feats is defined and contains the training features and labels
# total_idx = np.arange(len(train_feats))
total_idx = np.arange(len(train))
# np.random.shuffle(total_idx)

gc.collect()
# Cross-validation loop
from sklearn.model_selection import KFold, GroupKFold
#import tensorflow.keras.losses as tf_loss
all_oof = []
all_true = []
criterion = nn.KLDivLoss(reduction='batchmean')
#criterion = tf_loss.KLDivergence()
num_folds=4
for fold in range(2):
    # Split data into train and test sets for this fold
    train_idx, test_idx = train_test_split(total_idx, test_size=0.1)
    # test_idx = total_idx[fold * len(total_idx) // (num_folds):(fold + 1) * len(total_idx) // (Config.num_folds)]
    # train_idx = np.array([idx for idx in total_idx if idx not in test_idx])
    print('train length: ', len(train_idx), ' test length: ', len(test_idx))
    train_eegs = [eegs[i] for i in train_idx]
    test_eegs = [eegs[i] for i in test_idx]
    train_specs = [specs[i] for i in train_idx]
    test_specs = [specs[i] for i in test_idx]
    # keys = list(data.item().keys())
    # test_keys = [keys[i] for i in test_idx]
    # train_keys = [keys[i] for i in train_idx]
    train_dataset = CustomDataset(train.iloc[train_idx], train_eegs,train_specs)
    train_dataloader = DataLoader(train_dataset, batch_size=40, shuffle=True,  num_workers=0, drop_last=True)
    test_dataset = CustomDataset(train.iloc[test_idx], test_eegs,test_specs)
    test_dataloader = DataLoader(test_dataset, batch_size=40, shuffle=False,  num_workers=0, drop_last=True)
    torch.cuda.empty_cache()

    # Initialize EfficientNet-B0 model with pretrained weights
    #model = timm.create_model('convnext_base', pretrained=True,drop_rate=.2,drop_path_rate=.2 ,num_classes=1000, in_chans=1)
    #model1 = timm.create_model('tf_efficientnet_b3.ns_jft_in1k', in_chans=4, drop_path_rate=.3,num_classes=1000,pretrained=True)
    #model = model.eval()

    # get model specific transforms (normalization, resize)
    # data_config = timm.data.resolve_model_data_config(model)
    # transforms = timm.data.create_transform(**data_config, is_training=False)
    #model1 = redo_classifier(model1)
    # model2 = timm.create_model('tf_efficientnet_b3.ns_jft_in1k', in_chans=1, drop_path_rate=.2,num_classes=1000,pretrained=True)
    # model2 = redo_classifier(model2)
    model=CombinedModel()
    #model = nn.Sequential(nn.Conv2d(5,3,(1,1)),m1)
    model.to(device)
    

    optimizer = optim.NAdam(model.parameters(), lr=0.0002, weight_decay=0.02)
    scheduler = CosineAnnealingLR(optimizer, T_max=9)

    best_test_loss = float('inf')
    train_losses = []
    test_losses = []

    print(f"Starting training for fold {fold + 1}")

    # Training loop
    for epoch in range(2):
        model.train()
        train_loss = []

        count=0
        for x,x2,y in train_dataloader:
            count=count+1
            optimizer.zero_grad()

            x = x.permute(0,3,1,2)
            x=torch.nan_to_num(x)
            x2=torch.nan_to_num(x2)

            train_pred = model(x.to(device),x2.to(device))
            loss = KL_loss(y.to(device),train_pred)
           # loss2 = criterion(F.softmax(train_pred, 1), y.to(device))
            # loss = criterion(y,train_pred.detach().cpu().numpy())
            # loss = torch.tensor(loss.numpy(), requires_grad=True)
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss.append(loss.item())
            if count%30==0:
                print(loss.item())
                

        epoch_train_loss = np.mean(train_loss)
        train_losses.append(epoch_train_loss)
        print(f"Epoch {epoch + 1}: Train Loss = {epoch_train_loss:.2f}")

        scheduler.step()

        # Evaluation loop
        model.eval()
        test_loss = []
        with torch.no_grad():
            for x,x2, y in test_dataloader:
                x = x.permute(0,3,1,2)
                x=torch.nan_to_num(x)
                x2=torch.nan_to_num(x2)
                #x = x.unsqueeze(1)
                #x = transforms(x)

                test_pred = model(x.to(device), x2.to(device))
                loss = KL_loss(y.to(device),train_pred)
                #loss = criterion(F.log_softmax(test_pred, 1), y.to(device))
                #loss = criterion(F.log_softmax(train_pred, -1), y.to(device))
                # loss = criterion(test_pred.detach().cpu.numpy(),y)
                # loss = torch.tensor(loss.numpy())
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

    model.eval()
    test_preds = []
    with torch.no_grad():
        for x,x2,y in test_dataloader:
            x = x.permute(0,3,1,2)
            x=torch.nan_to_num(x)
            x2=torch.nan_to_num(x2)
            test_pred = model(x.float().to(device),x2.float().to(device))
            test_preds.append(F.log_softmax(test_pred,1).detach().cpu())
    #print(test_preds)
    
    test_preds=np.vstack(test_preds)
    #print(test_preds.shape)
    all_oof.append(test_preds)
    all_true.append(train.iloc[test_idx][TARGETS].values)
    
    #del model, oof
    gc.collect()
    
all_oof = np.concatenate(all_oof)
all_true = np.concatenate(all_true)

# import sys
# sys.path.append('/kaggle/input/kaggle-kl-div')
from kaggle_kl_div import score


oof = pd.DataFrame(all_oof.copy())
oof['id'] = np.arange(len(oof))

true = pd.DataFrame(all_true.copy())
true['id'] = np.arange(len(true))

cv = score(solution=true, submission=oof, row_id_column_name='id')
print('CV Score KL-Div for EfficientNetB2 =',cv)

