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

PATH_OR = "C:\\Users\\marcb\\OneDrive\\Desktop\\Kaggle_Competitions\\Kaggle-Competitions\\HMS-harmful-brain-activity\\"
test_eeg = PATH_OR+'hms-harmful-brain-activity-classification/train_eegs/'
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

READ_SPEC_FILES = False

# READ ALL SPECTROGRAMS
PATH = PATH_OR+'hms-harmful-brain-activity-classification/train_spectrograms/'
files = os.listdir(PATH)
print(f'There are {len(files)} spectrogram parquets')

if READ_SPEC_FILES:    
    spectrograms = {}
    for i,f in enumerate(files):
        if i%100==0: print(i,', ',end='')
        tmp = pd.read_parquet(f'{PATH}{f}')
        name = int(f.split('.')[0])
        spectrograms[name] = tmp.iloc[:,1:].values
else:
    spectrograms = np.load(PATH_OR+'brain-spectograms/specs.npy',allow_pickle=True).item()
    
    
READ_EEG_SPEC_FILES = False

if READ_EEG_SPEC_FILES:
    all_eegs = {}
    for i,e in enumerate(train.eeg_id.values):
        if i%100==0: print(i,', ',end='')
        x = np.load(f'/kaggle/input/brain-eeg-spectrograms/EEG_Spectrograms/{e}.npy')
        all_eegs[e] = x
else:
    all_eegs = np.load(PATH_OR+'brain-eeg-spectograms/eeg_specs.npy',allow_pickle=True).item()
    
    
import albumentations as albu
TARS = {'Seizure':0, 'LPD':1, 'GPD':2, 'LRDA':3, 'GRDA':4, 'Other':5}
TARS2 = {x:y for y,x in TARS.items()}

import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as albu
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, data, mode='train', specs=spectrograms, eeg_specs=all_eegs, augment=False):
        self.data = data
        self.mode = mode
        self.specs = specs
        self.eeg_specs = eeg_specs
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        if self.mode == 'test':
            r = 0
        else:
            r = int((row['min'] + row['max']) // 4)
    
        X_tensor = torch.zeros((128, 256, 8), dtype=torch.float32)
        y_tensor = torch.zeros(6, dtype=torch.float32)
    
        for k in range(4):
            # EXTRACT 300 ROWS OF SPECTROGRAM
            img = self.specs[row.spec_id][r:r+300, k*100:(k+1)*100].T
    
            # LOG TRANSFORM SPECTROGRAM
            img = np.clip(img, np.exp(-4), np.exp(8))
            img = np.log(img)
    
            # STANDARDIZE PER IMAGE
            ep = 1e-6
            m = np.nanmean(img.flatten())
            s = np.nanstd(img.flatten())
            img = (img - m) / (s + ep)
            img = np.nan_to_num(img, nan=0.0)
    
            # CROP TO 256 TIME STEPS
            X_tensor[14:-14, :, k] = torch.from_numpy(img[:, 22:-22] / 2.0)
    
        # EEG SPECTROGRAMS
        #print(X_tensor.shape)
        img = self.eeg_specs[row.eeg_id]
        X_tensor[:, :, 4:] = torch.tensor(img)
        #print(list(row[TARGETS].values))
    
        if self.mode != 'test':
            y_tensor = torch.tensor(list(row[TARGETS].values))
    
        if self.augment:
            X_tensor = self.__random_transform(X_tensor)
    
        return X_tensor, y_tensor

    def __random_transform(self, img):
        composition = albu.Compose([
            albu.HorizontalFlip(p=0.5),
            # albu.VerticalFlip(p=0.5),
            # albu.CoarseDropout(max_holes=8, max_height=32, max_width=32, fill_value=0, p=0.5),
        ])
        return composition(image=img)['image']

def get_dataloader(data, batch_size=32, shuffle=False, augment=False, mode='train',
                   specs=spectrograms, eeg_specs=all_eegs):
    dataset = CustomDataset(data, mode, specs, eeg_specs, augment)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    return dataloader
    
    
from sklearn.model_selection import train_test_split

total_idx = np.arange(len(train))
train_idx, val_idx = train_test_split(total_idx, test_size=0.2)
train_idx

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

import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()

        self.fc1 = nn.Linear(8*128*256, 2048)
        self.fc2 = nn.Linear(2048, 256)
        #self.fc3 = nn.Linear(1024, 256)
        self.fc4 = nn.Linear(256, 64)
        #self.fc5 = nn.Linear(64, 16)
        self.fc6 = nn.Linear(64, 6)

    def forward(self, x):
        # Run the first tensor through the first model
        b = x.shape[0]
        x = x.reshape(b,-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
       # x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
       # x = F.relu(self.fc5(x))
        x = self.fc6(x)

        return x
        
import torch
import torch.nn as nn
import torchvision.models as models

import torch
import torch.nn as nn
import torchvision.models as models

def build_model(USE_KAGGLE_SPECTROGRAMS=True, USE_EEG_SPECTROGRAMS=True):
    base_model = models.efficientnet_b0(pretrained=True)
    base_model.classifier = nn.Identity()

    model = nn.Sequential(
        nn.Conv2d(8, 3, kernel_size=1),  # Convert 8 channels to 3 channels
        base_model,
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(1792, 6),
        nn.Softmax(dim=1)
    )

    opt = torch.optim.AdamW(model.parameters(), lr=4e-4)
    loss = nn.KLDivLoss()

    return model, opt, loss

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
gkf = GroupKFold(n_splits=5)
#train=train.iloc[0:1800]
for i, (train_index, valid_index) in enumerate(gkf.split(train, train.target, train.patient_id)):  
    
    print('#'*25)
    print(f'### Fold {i+1}')
    print(len(valid_index))
    
    train_gen = get_dataloader(train.iloc[train_index], batch_size=32, shuffle=True)
    val_gen = get_dataloader(train.iloc[valid_index], batch_size=32, shuffle=False)
    
    print(f'### train size {len(train_index)}, valid size {len(valid_index)}')
    print('#'*25)
    # model = timm.create_model('tf_efficientnet_b0.ns_jft_in1k', in_chans=8, drop_path_rate=.1,num_classes=1000,pretrained=True)

    # model = redo_classifier(model)
    model = SimpleModel()
    #model,_,_=build_model()

    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.005)
    scheduler = CosineAnnealingLR(optimizer, T_max=20)

    best_test_loss = float('inf')
    train_losses = []
    test_losses = []

    for epoch in range(20):
        model.train()
        train_loss = []

        count=0
        for x,y in train_gen:
            count=count+1
            optimizer.zero_grad()

            x = x.permute(0,3,1,2)

            train_pred = model(x.to(device))
            #loss = criterion(train_pred,y.to(device))
            loss = criterion(F.log_softmax(train_pred, 1), y.to(device))
            # loss = criterion(y,train_pred.detach().cpu().numpy())
            # loss = torch.tensor(loss.numpy(), requires_grad=True)
            loss.backward()
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
            for x,y in val_gen:
                x = x.permute(0,3,1,2)
                #x = x.unsqueeze(1)
                #x = transforms(x)

                test_pred = model(x.to(device))
                loss = criterion(F.log_softmax(test_pred, 1), y.to(device))
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
            torch.save(model.state_dict(), f"efficientnet_b0_fold{i}.pth")

        gc.collect()

    print(f"Fold {i + 1} Best Test Loss: {best_test_loss:.2f}")

    model.eval()
    test_preds = []
    with torch.no_grad():
        for x,y in val_gen:
            #x = x.permute(0,3,1,2)
            test_pred = model(x.to(device))
            test_preds.append(F.softmax(test_pred).detach().cpu())
    #print(test_preds)
    
    test_preds=np.vstack(test_preds)
    #print(test_preds.shape)
    all_oof.append(test_preds)
    all_true.append(train.iloc[valid_index][TARGETS].values)
    
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

