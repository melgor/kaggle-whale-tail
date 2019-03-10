import os
import sys
import time 
import tqdm

import math
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.models as models

from torch.utils.data import TensorDataset, DataLoader,Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
import pretrainedmodels

import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)

# internal packages
from training import getDataLoader
from models.layers import NormLinear
from models.training import *



class NormLinear(nn.Module):
    def __init__(self, in_features, out_features, temperature = 0.05, temperature_trainable = False):
        super(NormLinear, self).__init__()

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        self.scale = 1 / temperature
        if temperature_trainable:
            self.scale = nn.Parameter(torch.Tensor(1))
            nn.init.constant_(self.scale, 1 / temperature)

    def forward(self, x):
        x_norm = F.normalize(x)
        w_norm = F.normalize(self.weight)
        cosine = F.linear(x_norm, w_norm, None)
        out = cosine * self.scale
        return out

resume   = sys.argv[1]
encoder  = sys.argv[2]
x_test   = sys.argv[3]

train_folder = "/home/blcv/CODE/Kaggle/humpback_short_blażej/data/processed/train_bb_fastai2/"
test_df     = "/home/blcv/CODE/Kaggle/humpback_whale_identification/data/processed/sample_submission.csv"
test_folder = "/home/blcv/CODE/Kaggle/humpback_short_blażej/data/processed/test_bb_fastai2/"
option_da = ['gray']# [] #


label_encoder = LabelEncoder()
label_encoder.classes_ = np.load(encoder)
# encode whale as integers
X_test = pd.read_csv(x_test)
val_loader = getDataLoader(X_test, train_folder, 'val', option_da = option_da, image_size = 224, batch_size = 64)

# model preparation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = 'se_resnext101_32x4d'
model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
model.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
model.last_linear = nn.Sequential(*[nn.LayerNorm(model.last_linear.in_features, elementwise_affine = False),
                                nn.AlphaDropout(p=0.1),
                            NormLinear(model.last_linear.in_features, 5004)])

model = model.to(device)
model = nn.DataParallel(model)                           

# load trained model
checkpoint = torch.load(resume)
model.load_state_dict(checkpoint['state_dict'])


criterion = nn.CrossEntropyLoss().cuda()

acc1 = validate(val_loader, model, criterion)

# predict data on test dataset
test_data = pd.read_csv(test_df)

test_loader = getDataLoader(test_data, test_folder, 'test', image_size = 224, batch_size = 64, option_da = option_da)
# Test
model.eval()

num_best_guess = 5
test_pred_paths, test_pred_names = [], []
list_pred = []
with torch.no_grad():
    for batch_idx, (inputs, paths, data_indices) in enumerate(test_loader):
        inputs  = inputs.to(device)
        outputs = model(inputs)
        _,best_values = torch.sort(outputs, dim=1)
        best_values   = best_values.cpu().numpy()
        best_values_shape  = best_values.shape
        best_value_decoder = label_encoder.inverse_transform(best_values.ravel())
        best_value_decoder = best_value_decoder.reshape(best_values_shape)[:,-num_best_guess:][:,::-1]
        test_pred_paths.append(paths)
        test_pred_names.append(best_value_decoder)
        list_pred.append(outputs.cpu().numpy())

list_pred  = np.vstack(list_pred)
np.save(f"{os.path.dirname(resume)}/list_pred.npy", list_pred)


test_pred_names = [" ".join(elem) for elem in np.concatenate(test_pred_names, axis = 0)]
submission = pd.DataFrame(columns = ['Image', 'Id'])
submission['Image'] = np.concatenate(test_pred_paths, axis = 0)
submission['Id'] = test_pred_names
submission.to_csv("se101_resnext_fastaibb.csv", index = False)
