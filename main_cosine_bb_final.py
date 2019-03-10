import os
import sys
import time 
import tqdm
import logging

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
from models.training import *
from models.adamw import AdamW
from models.layers import NormLinear, CosineMarginCrossEntropy



seed         = 42
torch.manual_seed(seed)
np.random.seed(seed)

oversample = 0
hard_mining = False
only_gray = True
batch_size = 16
image_size = 448
num_epochs = 40
best_acc1 = 0
best_acc5 = 0
best_loss_val = 0


models_path = sys.argv[1]
os.makedirs(models_path,  exist_ok=True)


name = f'val1_normlinear_adamw01_seed{seed}_gray{only_gray}'
check_filename = f"{models_path}/checkpoint_base_gray.pth.tar"

train_df     = "/root/humpback_short/data/processed/train.csv"
train_folder = "/root/humpback_short/data/processed/train_bb_fastai/"


FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger('kaggle')
hdlr = logging.FileHandler(f'{models_path}/{name}.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr) 
logger.setLevel(logging.INFO)
logger.info('Start Logging')



# encode whale as integers
data = pd.read_csv(train_df)

new_whale_df = data[data.Id == "new_whale"] # only new_whale dataset
train_df = data[~(data.Id == "new_whale")] # no new_whale dataset, used for training


train_df['label'], label_encoder = prepare_labels(train_df.Id)
np.save(f'{models_path}/{name}_classes.npy', label_encoder.classes_)


# get validation set where we have one example of each class
im_count = train_df.Id.value_counts()
im_count.name = 'sighting_count'
train_df = train_df.join(im_count, on='Id')
X_test = train_df.sample(frac=1, random_state = seed)[(train_df.Id != 'new_whale') & (train_df.sighting_count > 1)].groupby('label').first()
X_test['label'] = pd.to_numeric(X_test.index)

# Train on all images
X_train = train_df

sample_weights = np.ones(X_train.shape[0])
train_loader = getDataLoader(X_train, train_folder, 'train', batch_size = batch_size, image_size = image_size, \
                                                        train_weights = sample_weights, replacement = False, option_da = ['gray'] if only_gray else [])
val_loader   = getDataLoader(X_test, train_folder, 'val', batch_size = batch_size, image_size = image_size, option_da = ['gray'] if only_gray else [])

new_whale_df.to_csv(f"{models_path}/new_whale.csv")
X_test.to_csv(f"{models_path}/X_test_{name}.csv")
X_train.to_csv(f"{models_path}/X_train_{name}.csv")


# model preparation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = 'se_resnext101_32x4d'
model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
model.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
model.last_linear = nn.Sequential(*[nn.LayerNorm(model.last_linear.in_features, elementwise_affine = False),
                                    NormLinear(model.last_linear.in_features, 5004)])

model = model.to(device)
model = nn.DataParallel(model)


# train all layers
other_parameters = [param for name, param in model.module.named_parameters() if 'last_linear' not in name]
optimizer = AdamW(
    [
        {"params": model.module.last_linear.parameters(), "lr": 1e-3},
        {"params": other_parameters},
    ], 
    lr=1e-4, weight_decay = 0.01)    
    

best_loss_val = 100 
criterion = CosineMarginCrossEntropy().cuda()
exp_lr_scheduler = StepLR(optimizer, step_size=18, gamma=0.1)
for epoch in range(num_epochs):
    exp_lr_scheduler.step()
   
    
    # train for one epoch
    sample_weights = train(train_loader, model, criterion, optimizer, epoch, sample_weights, neptune_ctx)

    # evaluate on validation set
    acc1, acc5, loss_val = validate(val_loader, model, criterion)
    neptune_ctx.channel_send('val-acc1', acc1)
    neptune_ctx.channel_send('val-acc5', acc5)
    neptune_ctx.channel_send('val-loss', loss_val)
    neptune_ctx.channel_send('lr', float(exp_lr_scheduler.get_lr()[0]))
    
    logger.info(f'Epoch: {epoch} Acc1: {acc1} Acc5: {acc5} Val-Loss: {loss_val}')
    
    # remember best acc@1 and save checkpoint
    is_best = acc1 >= best_acc1     
    best_acc1 = max(acc1, best_acc1)
    save_checkpoint({
        'epoch': epoch + 1,
        'arch': 'resnet18',
        'state_dict': model.state_dict(),
        'best_acc1': best_acc1,
        }, is_best, name = name + "_acc1", filename = check_filename)
        

print(f"Best ACC: {best_acc1}")

