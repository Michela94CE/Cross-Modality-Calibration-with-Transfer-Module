import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

import scipy.io as sio
from scipy.stats import zscore
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import models, transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F

import os
import shutil

from random import choice, sample, seed, randint, random, gauss
from scipy.io import loadmat
import skimage.morphology as skm
from torch.utils.data import ConcatDataset, DataLoader, Dataset
import math
from mpl_toolkits.axes_grid1 import ImageGrid


torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


#clinical features
#HT:hormone therapy - menopausa: menopausal status - G:grading
Selected_features= ['age', 'familiarity', 'HT', 'menopausa','dimensions', 'ER','PgR', 'ki67','HER2', 'G' ]
IDX_TO_CONSIDER = [] #id of the selected features in the list


# Data Normalization
# several data normalization functions
def normalizeZscore(x, ax): 
  xz = zscore(x, axis=ax)
  return xz

def rangeNormalization(x, supLim, infLim): 
  x_norm = ( (x - np.min(x)) / (np.max(x)- np.min(x)) )*(supLim - infLim) + infLim
  assert np.min(x_norm) >= infLim
  assert np.max(x_norm) <= supLim
  return x_norm

def np_imadjust(x, q1,q2): 
  assert q1<q2
  assert q1+q2 == 1
  qq = np.quantile(x, [q1, q2])
  new = np.clip(x, qq[0], qq[1])
  return new

def multiply_mask(y, m):
  y_new = y*np.repeat(m[:,:,:,np.newaxis], 4, axis =3)
  return y_new

def convex_hull(mask): 
  xx,yy,zz = np.where(mask>0)
  zz_u = np.unique(zz)
  new_mask = np.zeros(mask.shape)
  for i in zz_u:
    new_mask[:,:,i] = skm.convex_hull_image(mask[:,:,i])
  return new_mask

def complete_convex_hull(mask):
  new_mask = convex_hull(mask) #x y z
  new_mask = new_mask.transpose(2,0,1)   #z x y

  new_mask = convex_hull(new_mask)
  new_mask = new_mask.transpose(0,2,1)  #z y x

  new_mask = convex_hull(new_mask)
  new_mask = new_mask.transpose(2,1,0)
  return new_mask

def getFilesForSubset(basepath, list_classes, include_patient):
  ListFiles=[]
  for c in list_classes:
    listofFiles = os.listdir(basepath + '/' + c)
    for file in listofFiles:
      if include_patient(basepath + '/' + c + '/' + file):
        ListFiles.append((basepath + '/' + c + '/' + file, list_classes.index(c)))
  return ListFiles

def getListOffiles(basepath, list_classes, classe, include_patient):
  ListFiles=[]
  listofFiles = os.listdir(basepath + '/' + classe)
  for file in listofFiles:
    if include_patient(basepath + '/' + classe + '/' + file):
      ListFiles.append((basepath + '/' + classe + '/' + file, list_classes.index(classe)))
  return ListFiles, len(ListFiles)

class FocalLoss(nn.modules.loss._WeightedLoss):
  def __init__(self, weight=None, gamma=2,reduction='mean'):
    super(FocalLoss, self).__init__(weight,reduction=reduction)
    self.gamma = gamma
    self.weight = weight
    self.reduction = reduction

  def forward(self, input, target):
    ce_loss = F.cross_entropy(input, target,reduction='none',weight=self.weight)
    pt = torch.exp(-ce_loss)
    focal_loss = ((1 - pt) ** self.gamma * ce_loss)
    if self.reduction == 'mean':
      focal_loss = focal_loss.mean()
    else:
      focal_loss = focal_loss.sum()
    return focal_loss

#-------------------------------------------------------------------------------> Data Augmentation
class ToTensor3D(torch.nn.Module):  
  def __init__(self):
    super().__init__()

  def forward(self, tensor):
    y_new = torch.from_numpy(tensor.transpose(3,2,0,1))
    return y_new

  def __repr__(self):
    return self.__class__.__name__ + '()'

class DeleteMask(torch.nn.Module): 
  def __init__(self):
    super().__init__()

  def forward(self, tensor):
    return tensor

  def __repr__(self):
    return self.__class__.__name__ + '()'

class Resize3D(torch.nn.Module): 
  def __init__(self, size=(32,32,32)):
    self.size = size
    super().__init__()

  def forward(self, tensor):
    #print(tensor.shape)
    #print(tensor.unsqueeze(0).shape)
    img = F.interpolate( tensor.unsqueeze(0), self.size, align_corners =True, mode='trilinear').squeeze(0)
    #print(img.shape)
    return img

  def __repr__(self):
    return self.__class__.__name__ + '(size={})'.format(self.size)


class Random_Rotation(torch.nn.Module):
  def __init__(self, p=0.5, n=1):
    self.p = p                       
    self.n = n                        
    super().__init__()

  def forward(self, img):
    if random() < self.p:
      img = torch.rot90(img,self.n,dims=[2,3])
    return img

  def __repr__(self):
    return self.__class__.__name__ + '(p={}, n={})'.format(self.p, self.n)

#------------------------------
class RandomZFlip(torch.nn.Module):
  def __init__(self, p=0.5):
    self.p = p                      
    super().__init__()

  def forward(self, img):

    if random() < self.p:
      img = torch.flip(img, [1])
    return img

  def __repr__(self):
    return self.__class__.__name__ + '(p={})'.format(self.p)

#------------------------------ 
class Normalize(torch.nn.Module):
  def __init__(self, mean, std):
    self.mean = mean
    self.std = std
    super().__init__()

  def forward(self, tensor):
    app =  tensor[0,:,:,:]
    new = ((app - self.mean[0]) /self.std[0]).unsqueeze(0)

    for i in range(1, tensor.shape[0]):
      app =  tensor[i,:,:,:]
      app = (app - self.mean[i]) /self.std[i]
      new = torch.cat([new, app.unsqueeze(0)], dim=0)
    return new

  def __repr__(self):
    return self.__class__.__name__ + '(mean={}, std={})'.format(self.mean, self.std )

#------------------------------ 
class NormalizeOneMeanStd(torch.nn.Module):
  def __init__(self, mean, std):
    self.mean = mean
    self.std = std
    super().__init__()

  def forward(self, tensor):
    new = (tensor - self.mean.item())/self.std.item()
    return new

  def __repr__(self):
    return self.__class__.__name__ + '(mean={}, std={})'.format(self.mean, self.std)

#------------------------------ Data Balancing
class BalanceConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        l = max([len(dataset) for dataset in datasets])
        for dataset in datasets:
            while len(dataset) < l:
                dataset.samples += sample(dataset.samples, min(len(dataset), l - len(dataset)))
        super(BalanceConcatDataset, self).__init__(datasets)

#-------------------------------------> Media e STD  -> numpy
def np_computeMeanAndStd_all_element(data, channel_sum, channel_sqared_sum, num_batches):
  ss = np.array(data.shape)
  channel_sum += np.sum(data)
  channel_sqared_sum  += np.sum(data**2)
  num_batches += ss.prod()  #devo fare il prodotto delle dimensioni
  return channel_sum, channel_sqared_sum, num_batches



def np_computeMeanAndStd_all(train_files):  #-> specific for the dataset
  channel_sum_dce, channel_sqared_sum_dce = 0.0,0.0
  num_batches_dce = 0
  channel_sum_wat, channel_sqared_sum_wat = 0.0,0.0
  num_batches_wat = 0
  channel_sum_dwi, channel_sqared_sum_dwi = 0.0,0.0
  num_batches_dwi = 0

  for f,l in train_files:
    data_dce, data_water, data_dwi, cliniche = readVolume(f)
    channel_sum_dce, channel_sqared_sum_dce, num_batches_dce = np_computeMeanAndStd_all_element(data_dce, channel_sum_dce, channel_sqared_sum_dce, num_batches_dce)
    channel_sum_wat, channel_sqared_sum_wat, num_batches_wat = np_computeMeanAndStd_all_element(data_water, channel_sum_wat, channel_sqared_sum_wat, num_batches_wat)
    channel_sum_dwi, channel_sqared_sum_dwi, num_batches_dwi = np_computeMeanAndStd_all_element(data_dwi, channel_sum_dwi, channel_sqared_sum_dwi, num_batches_dwi)

  mean_dce =channel_sum_dce/num_batches_dce
  std_dce = (channel_sqared_sum_dce/num_batches_dce - mean_dce**2)**0.5
  mean_wat =channel_sum_wat/num_batches_wat
  std_wat = (channel_sqared_sum_wat/num_batches_wat - mean_wat**2)**0.5
  mean_dwi =channel_sum_dwi/num_batches_dwi
  std_dwi = (channel_sqared_sum_dwi/num_batches_dwi - mean_dwi**2)**0.5
  return mean_dce, std_dce, mean_wat, std_wat, mean_dwi, std_dwi

#------------------------------------->
def np_computeMeanAndStd_channnel_elemet(data, channel_sum, channel_sqared_sum, num_batches):
  ss = np.array(data.shape)
  channel_sum += np.sum(data, axis=tuple(range(0,ss.shape[0]-1)))
  channel_sqared_sum  += np.sum(data**2, axis=tuple(range(0,ss.shape[0]-1)))
  num_batches += ss[:ss.shape[0]-1].prod()
  return channel_sum, channel_sqared_sum, num_batches

def np_computeMeanAndStd_channnel(train_files): #specific for the dataset
  channel_sum_dce, channel_sqared_sum_dce = 0.0,0.0
  num_batches_dce = 0
  channel_sum_wat, channel_sqared_sum_wat = 0.0,0.0
  num_batches_wat = 0
  channel_sum_dwi, channel_sqared_sum_dwi = 0.0,0.0
  num_batches_dwi = 0

  for f,l in train_files:
    data_dce, data_water, data_dwi, cliniche = readVolume(f)
    channel_sum_dce, channel_sqared_sum_dce, num_batches_dce = np_computeMeanAndStd_channnel_elemet(data_dce, channel_sum_dce, channel_sqared_sum_dce, num_batches_dce)
    channel_sum_wat, channel_sqared_sum_wat, num_batches_wat = np_computeMeanAndStd_channnel_elemet(data_water, channel_sum_wat, channel_sqared_sum_wat, num_batches_wat)
    channel_sum_dwi, channel_sqared_sum_dwi, num_batches_dwi = np_computeMeanAndStd_channnel_elemet(data_dwi, channel_sum_dwi, channel_sqared_sum_dwi, num_batches_dwi)

  mean_dce =channel_sum_dce/num_batches_dce
  std_dce = (channel_sqared_sum_dce/num_batches_dce - mean_dce**2)**0.5
  mean_wat =channel_sum_wat/num_batches_wat
  std_wat = (channel_sqared_sum_wat/num_batches_wat - mean_wat**2)**0.5
  mean_dwi =channel_sum_dwi/num_batches_dwi
  std_dwi = (channel_sqared_sum_dwi/num_batches_dwi - mean_dwi**2)**0.5
  return mean_dce, std_dce, mean_wat, std_wat, mean_dwi, std_dwi

#-------------------------------------> Media e STD  -> torch
def torch_computeMeanAndStd_all_element(data, channel_sum, channel_sqared_sum, num_batches):
  data = torch.from_numpy(data)
  ss = torch.tensor(data.shape)
  channel_sum += torch.sum(data)
  channel_sqared_sum  += torch.sum(data**2)
  num_batches += ss.prod()
  return channel_sum, channel_sqared_sum, num_batches

def torch_computeMeanAndStd_all(train_files): #-> specific for the dataset
  channel_sum_dce, channel_sqared_sum_dce = 0.0,0.0
  num_batches_dce = 0
  channel_sum_wat, channel_sqared_sum_wat = 0.0,0.0
  num_batches_wat = 0
  channel_sum_dwi, channel_sqared_sum_dwi = 0.0,0.0
  num_batches_dwi = 0

  for f,l in train_files:
    data_dce, data_water, data_dwi, cliniche = readVolume(f)
    channel_sum_dce, channel_sqared_sum_dce, num_batches_dce = torch_computeMeanAndStd_all_element(data_dce, channel_sum_dce, channel_sqared_sum_dce, num_batches_dce)
    channel_sum_wat, channel_sqared_sum_wat, num_batches_wat = torch_computeMeanAndStd_all_element(data_water, channel_sum_wat, channel_sqared_sum_wat, num_batches_wat)
    channel_sum_dwi, channel_sqared_sum_dwi, num_batches_dwi = torch_computeMeanAndStd_all_element(data_dwi, channel_sum_dwi, channel_sqared_sum_dwi, num_batches_dwi)

  mean_dce =channel_sum_dce/num_batches_dce
  std_dce = (channel_sqared_sum_dce/num_batches_dce - mean_dce**2)**0.5
  mean_wat =channel_sum_wat/num_batches_wat
  std_wat = (channel_sqared_sum_wat/num_batches_wat - mean_wat**2)**0.5
  mean_dwi =channel_sum_dwi/num_batches_dwi
  std_dwi = (channel_sqared_sum_dwi/num_batches_dwi - mean_dwi**2)**0.5
  return mean_dce, std_dce, mean_wat, std_wat, mean_dwi, std_dwi

def torch_computeMeanAndStd_channnel_element(data, channel_sum, channel_sqared_sum, num_batches):
    data = torch.from_numpy(data)
    ss = torch.tensor(data.shape)
    channel_sum += torch.sum(data, dim=list(range(0, len(ss)-1)) )
    channel_sqared_sum  += torch.sum(data**2,  dim=list(range(0, len(ss)-1)))
    num_batches += ss[0:len(ss)-1].prod()
    return channel_sum, channel_sqared_sum, num_batches

def torch_computeMeanAndStd_channnel(train_files): #specific for the dataset
  channel_sum_dce, channel_sqared_sum_dce = 0.0,0.0
  num_batches_dce = 0
  channel_sum_wat, channel_sqared_sum_wat = 0.0,0.0
  num_batches_wat = 0
  channel_sum_dwi, channel_sqared_sum_dwi = 0.0,0.0
  num_batches_dwi = 0

  for f,l in train_files:
    data_dce, data_water, data_dwi, cliniche = readVolume(f)
    channel_sum_dce, channel_sqared_sum_dce, num_batches_dce = torch_computeMeanAndStd_channnel_element(data_dce, channel_sum_dce, channel_sqared_sum_dce, num_batches_dce)
    channel_sum_wat, channel_sqared_sum_wat, num_batches_wat = torch_computeMeanAndStd_channnel_element(data_water, channel_sum_wat, channel_sqared_sum_wat, num_batches_wat)
    channel_sum_dwi, channel_sqared_sum_dwi, num_batches_dwi = torch_computeMeanAndStd_channnel_element(data_dwi, channel_sum_dwi, channel_sqared_sum_dwi, num_batches_dwi)

  mean_dce =channel_sum_dce/num_batches_dce
  std_dce = (channel_sqared_sum_dce/num_batches_dce - mean_dce**2)**0.5
  mean_wat =channel_sum_wat/num_batches_wat
  std_wat = (channel_sqared_sum_wat/num_batches_wat - mean_wat**2)**0.5
  mean_dwi =channel_sum_dwi/num_batches_dwi
  std_dwi = (channel_sqared_sum_dwi/num_batches_dwi - mean_dwi**2)**0.5
  return mean_dce, std_dce, mean_wat, std_wat, mean_dwi, std_dwi

##-------------------------------------------------------------------------------> Training functions
# function to train the network using a validation set
def train_loop_validation(model_conv,
                          trainset, Val, test,
                          start, num_epoch,
                          loader_opts,
                          criterionCNN, optimizer_conv,
                          best_acc, best_loss, best_epoca,
                          outputPath):

  for epochs in range(start, num_epoch + 1):

    TrainLoader = DataLoader(trainset, shuffle=True, **loader_opts)

    modelLoss_train = 0.0
    modelAcc_train = 0.0
    totalSize = 0

    model_conv.train()
    totPred = torch.empty(0)
    totLabels = torch.empty(0)

    #-----------------------------------------------------------------------------> TRAIN
    for inputs_dce, inputs_water, inputs_dwi, inputs_cliniche, labels in TrainLoader:
      inputs_dce = inputs_dce.type(torch.FloatTensor).cuda()
      inputs_water = inputs_water.type(torch.FloatTensor).cuda()
      inputs_dwi = inputs_dwi.type(torch.FloatTensor).cuda()
      inputs_cliniche = inputs_cliniche.type(torch.FloatTensor).cuda()
      labels = labels.cuda()

      optimizer_conv.zero_grad()
      model_conv.zero_grad()

      y = model_conv(inputs_dce, inputs_water, inputs_dwi, inputs_cliniche)
      outp, preds = torch.max(y, 1)
      lossCNN = criterionCNN(y, labels) #media per batch

      lossCNN.backward()
      optimizer_conv.step()

      totPred = torch.cat((totPred, preds.cpu()))
      totLabels = torch.cat((totLabels, labels.cpu()))

      modelLoss_train += lossCNN.item() * inputs_dce.size(0)
      totalSize += inputs_dce.size(0)
      modelAcc_train += torch.sum(preds == labels.data).item()


    modelLoss_epoch_train = modelLoss_train/totalSize
    modelAcc_epoch_train  = modelAcc_train/totalSize

    totPred = totPred.numpy()
    totLabels = totLabels.numpy()
    acc = np.sum((totPred == totLabels).astype(int))/totalSize

    x = totLabels[np.where(totLabels == 1)]
    y = totPred[np.where(totLabels == 1)]
    acc_1_T = np.sum((x == y).astype(int))/x.shape[0]

    x = totLabels[np.where(totLabels == 0)]
    y = totPred[np.where(totLabels == 0)]
    acc_0_T = np.sum((x == y).astype(int))/y.shape[0]

    with open(outputPath + 'lossTrain.txt', "a") as file_object:
      file_object.write(str(modelLoss_epoch_train) +'\n')
    with open(outputPath + 'AccTrain.txt', "a") as file_object:
      file_object.write(str(modelAcc_epoch_train)+'\n')

    torch.save(model_conv.state_dict(), outputPath + 'train_weights.pth')

    #-----------------------------------------------------------------------------> VALIDATION

    model_conv.eval()

    totalSize_val = 0
    modelLoss_val = 0.0
    modelAcc_val = 0.0

    totPred_val = torch.empty(0)
    totLabels_val = torch.empty(0)

    ValLoader = DataLoader(Val, shuffle=True, **loader_opts)
    for inputs_dce, inputs_water, inputs_dwi, inputs_cliniche, labels in ValLoader:
      inputs_dce = inputs_dce.type(torch.FloatTensor).cuda()
      inputs_water = inputs_water.type(torch.FloatTensor).cuda()
      inputs_dwi = inputs_dwi.type(torch.FloatTensor).cuda()
      inputs_cliniche = inputs_cliniche.type(torch.FloatTensor).cuda()
      labels = labels.cuda()

      y = model_conv(inputs_dce, inputs_water, inputs_dwi, inputs_cliniche)
      outp, preds = torch.max(y, 1)
      lossCNN = criterionCNN(y, labels)

      totPred_val = torch.cat((totPred_val, preds.cpu()))
      totLabels_val = torch.cat((totLabels_val, labels.cpu()))

      modelLoss_val += lossCNN.item() * inputs_dce.size(0)  #Non pesata -> semplice media
      totalSize_val += inputs_dce.size(0)
      modelAcc_val += torch.sum(preds == labels.data).item()


    modelLoss_epoch_val = modelLoss_val/totalSize_val
    modelAcc_epoch_val = modelAcc_val/totalSize_val

    totPred_val = totPred_val.numpy()
    totLabels_val = totLabels_val.numpy()
    acc_val = np.sum((totPred_val == totLabels_val).astype(int))/totalSize_val

    x = totLabels_val[np.where(totLabels_val == 1)]
    y = totPred_val[np.where(totLabels_val == 1)]
    acc_1_V = np.sum((x == y).astype(int))/x.shape[0]

    x = totLabels_val[np.where(totLabels_val == 0)]
    y = totPred_val[np.where(totLabels_val == 0)]
    acc_0_v = np.sum((x == y).astype(int))/y.shape[0]


    with open(outputPath + 'lossVal.txt', "a") as file_object:
      file_object.write(str(modelLoss_epoch_val) +'\n')

    with open(outputPath + 'AccVal.txt', "a") as file_object:
      file_object.write(str(modelAcc_epoch_val)+'\n')

    with open(outputPath + 'AccVal_0.txt', "a") as file_object:
      file_object.write(str(acc_0_v)+'\n')

    with open(outputPath + 'AccVal_1.txt', "a") as file_object:
      file_object.write(str(acc_1_V)+'\n')

    print('[Epoch %d][TRAIN on %d [Loss: %.4f - ACC_T: %.4f - ACC_0: %.4f - ACC_1: %.4f ]][VAL on %d [Loss: %.4f - ACC_T: %.4f - ACC_0: %.4f - ACC_1: %.4f]]'
          %(epochs, totalSize, modelLoss_epoch_train, modelAcc_epoch_train, acc_0_T, acc_1_T,
            totalSize_val, modelLoss_epoch_val,
            modelAcc_epoch_val, acc_0_v, acc_1_V))

    if epochs == 1 or (modelLoss_epoch_val <= best_loss) :

      print('     .... Saving best weights ....')
      best_acc = modelAcc_epoch_val
      best_loss = modelLoss_epoch_val
      best_epoca = epochs

      #salvataggio dei migliori pesi sul validation
      torch.save(model_conv.state_dict(), outputPath + 'best_model_weights.pth')

      #vedi il test come va
      tot_size_test = 0
      model_loss_test = 0.0
      modelAcc_acc_test = 0.0
      totPred_test = torch.empty(0)
      totLabels_test = torch.empty(0)
      
      #check with the best weights [this part can be removed]
      TestLoader = DataLoader(test, shuffle=True, **loader_opts)

      for  inputs_dce, inputs_water, inputs_dwi, inputs_cliniche, labels in TestLoader:
        inputs_dce = inputs_dce.type(torch.FloatTensor).cuda()
        inputs_water = inputs_water.type(torch.FloatTensor).cuda()
        inputs_dwi = inputs_dwi.type(torch.FloatTensor).cuda()
        inputs_cliniche = inputs_cliniche.type(torch.FloatTensor).cuda()
        labels = labels.cuda()

        y = model_conv(inputs_dce, inputs_water, inputs_dwi, inputs_cliniche)
        outp, preds = torch.max(y, 1)
        lossCNN = criterionCNN(y, labels)

        totPred_test = torch.cat((totPred_test, preds.cpu()))
        totLabels_test = torch.cat((totLabels_test, labels.cpu()))

        model_loss_test += lossCNN.item() * inputs_dce.size(0)  #Non pesata -> semplice media
        tot_size_test += inputs_dce.size(0)
        modelAcc_acc_test += torch.sum(preds == labels.data).item()

      modelLoss_epoch_test = model_loss_test/tot_size_test
      modelAcc_epoch_test = modelAcc_acc_test/tot_size_test

      totPred_test = totPred_test.numpy()
      totLabels_test = totLabels_test.numpy()
      acc_val = np.sum((totPred_test == totLabels_test).astype(int))/tot_size_test

      x = totLabels_test[np.where(totLabels_test == 1)]
      y = totPred_test[np.where(totLabels_test == 1)]
      acc_1_test = np.sum((x == y).astype(int))/x.shape[0]

      x = totLabels_test[np.where(totLabels_test == 0)]
      y = totPred_test[np.where(totLabels_test == 0)]
      acc_0_test = np.sum((x == y).astype(int))/y.shape[0]


      print('      [TEST on %d [Loss: %.4f - ACC_T: %.4f - ACC_0: %.4f - ACC_1: %.4f ]]'
            %(tot_size_test, modelLoss_epoch_test, modelAcc_epoch_test, acc_0_test, acc_1_test))


    sio.savemat(outputPath + 'check_point.mat', {'best_acc': best_acc,
                                                 'best_loss': best_loss,
                                                 'best_epoca': best_epoca,
                                                 'last_epoch': epochs})
  return model_conv

## #-------------------------------------------------------------------------------> Predict function
def prediction_on_Test(model_conv, test, transform):
  func = nn.Softmax(dim=1)
  predicted = pd.DataFrame()
  testFiles = test.samples

  for path, label_true in testFiles:
    inputs_dce, inputs_water, inputs_dwi, inputs_cliniche= readVolume(path)

    if transform is not None:
      inputs_dce = transform[0](inputs_dce)
      inputs_water = transform[1](inputs_water)
      inputs_dwi = transform[2](inputs_dwi)
      inputs_cliniche = torch.from_numpy(inputs_cliniche)

    inputs_dce = inputs_dce.type(torch.FloatTensor).unsqueeze(0).cuda()
    inputs_water = inputs_water.type(torch.FloatTensor).unsqueeze(0).cuda()
    inputs_dwi = inputs_dwi.type(torch.FloatTensor).unsqueeze(0).cuda()
    inputs_cliniche = inputs_cliniche.type(torch.FloatTensor).unsqueeze(0).cuda()

    y = model_conv(inputs_dce, inputs_water, inputs_dwi, inputs_cliniche)
    outp, preds = torch.max(y, 1)
    y = func(y)

    for i in range(0, inputs_dce.shape[0]):
      #add the information you need in the analysis
      predicted = predicted.append({'filename': path.split('/')[-1],
                                    'prob0': y[i,0].item(),
                                    'prob1': y[i,1].item(),
                                    'predicted': preds[i].item(),
                                    'true_class': label_true,
                                    }, ignore_index=True)
  return predicted
## #------------------------------------------------------------------------------->Retrain function
def train_loop(model_conv,
               trainset, test,
               startEpoch,
               num_epoch, loader_opts,
               criterionCNN, optimizer_conv, outputPath,
               weightName, chekpointName):

  for epochs in range(startEpoch, num_epoch + 1):

    TrainLoader = DataLoader(trainset, shuffle=True, **loader_opts)
    modelLoss_train = 0.0
    modelAcc_train = 0.0
    totalSize = 0

    model_conv.train()
    totPred = torch.empty(0)
    totLabels = torch.empty(0)
    #-----------------------------------------------------------------------------> TRAIN
    for inputs_dce, inputs_water, inputs_dwi, inputs_cliniche, labels in TrainLoader:
      inputs_dce = inputs_dce.type(torch.FloatTensor).cuda()
      inputs_water = inputs_water.type(torch.FloatTensor).cuda()
      inputs_dwi = inputs_dwi.type(torch.FloatTensor).cuda()
      inputs_cliniche = inputs_cliniche.type(torch.FloatTensor).cuda()
      labels = labels.cuda()

      optimizer_conv.zero_grad()
      model_conv.zero_grad()

      y = model_conv(inputs_dce, inputs_water, inputs_dwi, inputs_cliniche)
      outp, preds = torch.max(y, 1)
      lossCNN = criterionCNN(y, labels) #media per batch

      lossCNN.backward()
      optimizer_conv.step()

      totPred = torch.cat((totPred, preds.cpu()))
      totLabels = torch.cat((totLabels, labels.cpu()))

      modelLoss_train += lossCNN.item() * inputs_dce.size(0)
      totalSize += inputs_dce.size(0)
      modelAcc_train += torch.sum(preds == labels.data).item()

    modelLoss_epoch_train = modelLoss_train/totalSize
    modelAcc_epoch_train  = modelAcc_train/totalSize

    totPred = totPred.numpy()
    totLabels = totLabels.numpy()
    acc = np.sum((totPred == totLabels).astype(int))/totalSize

    x = totLabels[np.where(totLabels == 1)]
    y = totPred[np.where(totLabels == 1)]
    acc_1_T = np.sum((x == y).astype(int))/x.shape[0]

    x = totLabels[np.where(totLabels == 0)]
    y = totPred[np.where(totLabels == 0)]
    acc_0_T = np.sum((x == y).astype(int))/y.shape[0]


    torch.save(model_conv.state_dict(), outputPath + weightName)
    sio.savemat(outputPath + chekpointName, {'last_epoch': epochs})

    #-----------------------------------------------------------------------------> VALIDATION
    model_conv.eval()
    tot_size_test = 0
    model_loss_test = 0.0
    modelAcc_acc_test = 0.0
    totPred_test = torch.empty(0)
    totLabels_test = torch.empty(0)

    TestLoader = DataLoader(test, shuffle=True, **loader_opts)

    for  inputs_dce, inputs_water, inputs_dwi, inputs_cliniche, labels in TestLoader:
      inputs_dce = inputs_dce.type(torch.FloatTensor).cuda()
      inputs_water = inputs_water.type(torch.FloatTensor).cuda()
      inputs_dwi = inputs_dwi.type(torch.FloatTensor).cuda()
      inputs_cliniche = inputs_cliniche.type(torch.FloatTensor).cuda()
      labels = labels.cuda()

      y = model_conv(inputs_dce, inputs_water, inputs_dwi, inputs_cliniche)
      outp, preds = torch.max(y, 1)
      lossCNN = criterionCNN(y, labels) #media per batch

      totPred_test = torch.cat((totPred_test, preds.cpu()))
      totLabels_test = torch.cat((totLabels_test, labels.cpu()))

      model_loss_test += lossCNN.item() * inputs_dce.size(0)  #Non pesata -> semplice media
      tot_size_test += inputs_dce.size(0)
      modelAcc_acc_test += torch.sum(preds == labels.data).item()

    modelLoss_epoch_test = model_loss_test/tot_size_test
    modelAcc_epoch_test = modelAcc_acc_test/tot_size_test

    totPred_test = totPred_test.numpy()
    totLabels_test = totLabels_test.numpy()
    acc_val = np.sum((totPred_test == totLabels_test).astype(int))/tot_size_test

    x = totLabels_test[np.where(totLabels_test == 1)]
    y = totPred_test[np.where(totLabels_test == 1)]
    acc_1_test = np.sum((x == y).astype(int))/x.shape[0]

    x = totLabels_test[np.where(totLabels_test == 0)]
    y = totPred_test[np.where(totLabels_test == 0)]
    acc_0_test = np.sum((x == y).astype(int))/y.shape[0]


    print('[Epoch %d][TRAIN on %d [Loss: %.4f  ACC: %.4f - ACC_0: %.4f - ACC_1: %.4f]][TEST on %d [Loss: %.4f ][ACC_T: %.4f - ACC_0: %.4f - ACC_1: %.4f]]'
          %(epochs, totalSize, modelLoss_epoch_train, modelAcc_epoch_train, acc_0_T, acc_1_T,
            tot_size_test, modelLoss_epoch_test,
            modelAcc_epoch_test, acc_0_test, acc_1_test))


  return model_conv

#Definizione trasformazioni
def definisciTransf(meant_dce, stdt_dce, meant_water, stdt_water, meant_dwi, stdt_dwi, tipoNormalizzazione_str):
  print(tipoNormalizzazione_str)
  choice = transforms.RandomChoice([Random_Rotation(p=0.5, n=1),
                                    Random_Rotation(p=0.5, n=2),
                                    Random_Rotation(p=0.5, n=3)])

  if tipoNormalizzazione_str == 'torch_computeMeanAndStd_all':
    NormFunction_dce = NormalizeOneMeanStd(meant_dce, stdt_dce)
    NormFunction_water = NormalizeOneMeanStd(meant_water, stdt_water)
    NormFunction_dwi = NormalizeOneMeanStd(meant_dwi, stdt_dwi)
  elif tipoNormalizzazione_str == 'torch_computeMeanAndStd_channnel':
    NormFunction_dce = Normalize(meant_dce, stdt_dce)
    NormFunction_water = Normalize(meant_water, stdt_water)
    NormFunction_dwi = Normalize(meant_dwi, stdt_dwi)
  else:
    NormFunction_dce = DeleteMask()
    NormFunction_water = DeleteMask()
    NormFunction_dwi =  DeleteMask()

  #----------------------------------------------------------------------------> CONTROLLO SULLA MASCHERA
  train_transform_dce = transforms.Compose([ToTensor3D(),
                                            NormFunction_dce,

                                            Resize3D(size=(64, 64, 64)),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomVerticalFlip(),
                                            RandomZFlip(),
                                            transforms.RandomRotation(degrees = 90),
                                            choice
                                            ])

  train_transform_water = transforms.Compose([ToTensor3D(),
                                            NormFunction_water,

                                            Resize3D(size=(64, 64, 64)),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomVerticalFlip(),
                                            RandomZFlip(),
                                            transforms.RandomRotation(degrees = 90),
                                            choice
                                            ])

  train_transform_dwi = transforms.Compose([ToTensor3D(),
                                            NormFunction_dwi,

                                            Resize3D(size=(64, 64, 64)),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomVerticalFlip(),
                                            RandomZFlip(),
                                            transforms.RandomRotation(degrees = 90),
                                            choice
                                            ])

  None_transform_dce  = transforms.Compose([ToTensor3D(),
                                            NormFunction_dce,
                                            Resize3D(size=(64, 64, 64))])

  None_transform_water  = transforms.Compose([ToTensor3D(),
                                            NormFunction_water,
                                            Resize3D(size=(64, 64, 64))])

  None_transform_dwi  = transforms.Compose([ToTensor3D(),
                                            NormFunction_dwi,
                                            Resize3D(size=(64, 64, 64))])

  return [train_transform_dce, train_transform_water, train_transform_dwi],  [None_transform_dce, None_transform_water, None_transform_dwi]

def statistiche(listFiles):
  mean_dce_a, std_dce_a, mean_wat_a, std_wat_a, mean_dwi_a, std_dwi_a = np_computeMeanAndStd_all(listFiles)
  print('DCE -->')
  print(mean_dce_a)
  print(std_dce_a)
  print('WATER -->')
  print(mean_wat_a)
  print(std_wat_a)
  print('DWI -->')
  print(mean_dwi_a)
  print(std_dwi_a)
  mean_dce_c, std_dce_c, mean_wat_c, std_wat_c, mean_dwi_c, std_dwi_c = np_computeMeanAndStd_channnel(listFiles)
  print('DCE -->')
  print(mean_dce_c)
  print(std_dce_c)
  print('WATER -->')
  print(mean_wat_c)
  print(std_wat_c)
  print('DWI -->')
  print(mean_dwi_c)
  print(std_dwi_c)
  mean_dce_a_t, std_dce_a_t, mean_wat_a_t, std_wat_a_t, mean_dwi_a_t, std_dwi_a_t = torch_computeMeanAndStd_all(listFiles)
  print('DCE -->')
  print(mean_dce_a_t)
  print(std_dce_a_t)
  print('WATER -->')
  print(mean_wat_a_t)
  print(std_wat_a_t)
  print('DWI -->')
  print(mean_dwi_a_t)
  print(std_dwi_a_t)
  mean_dce_c_t, std_dce_c_t, mean_wat_c_t, std_wat_c_t, mean_dwi_c_t, std_dwi_c_t = torch_computeMeanAndStd_channnel(listFiles)
  print('DCE -->')
  print(mean_dce_c_t)
  print(std_dce_c_t)
  print('WATER -->')
  print(mean_wat_c_t)
  print(std_wat_c_t)
  print('DWI -->')
  print(mean_dwi_c_t)
  print(std_dwi_c_t)
  return mean_dce_a_t, std_dce_a_t, mean_wat_a_t, std_wat_a_t, mean_dwi_a_t, std_dwi_a_t, mean_dce_c_t, std_dce_c_t, mean_wat_c_t, std_wat_c_t, mean_dwi_c_t, std_dwi_c_t
  
def readVolume(path):
  x = loadmat(path)
  y_new_dce= x['dce_volume'].astype(np.float32)

  y_new_water= x['water_volume'].astype(np.float32)
  y_new_water = y_new_water[:,:,:,np.newaxis]

  y_new_dwi= x['dwi_volume'].astype(np.float32)
  y_new_dwi = y_new_dwi[:,:,:,np.newaxis]

  y_new_dce = rangeNormalization(y_new_dce, 1, 0)
  y_new_water = rangeNormalization(y_new_water, 1, 0)
  y_new_dwi = rangeNormalization(y_new_dwi, 1, 0)

  y_cliniche = x['features'][0,IDX_TO_CONSIDER]

  return y_new_dce, y_new_water, y_new_dwi, y_cliniche


class My_DatasetFolder(Dataset):
  def __init__(self, root,  transform, is_valid_file, list_classes):
    self.root = root
    self.transform = transform
    self.is_valid_file = is_valid_file
    self.list_classes = list_classes
    self.samples = self.__get_samples()

  def __len__(self):
    return len(self.samples)

  def __get_samples(self):
    ListFiles=[]
    for c in self.list_classes:
      listofFiles = os.listdir(self.root + '/' + c)
      for file in listofFiles:
        if self.is_valid_file(self.root + '/' + c + '/' + file):
          ListFiles.append((self.root + '/' + c + '/' + file, self.list_classes.index(c)))
    return ListFiles

  def __getitem__(self, index: int):
    path, target = self.samples[index]
    sample_dce, sample_water, sample_dwi, sample_cliniche = readVolume(path)
    if self.transform is not None:
      sample_dce = self.transform[0](sample_dce)
      sample_water = self.transform[1](sample_water)
      sample_dwi = self.transform[2](sample_dwi)
      sample_cliniche = torch.from_numpy(sample_cliniche)

    return sample_dce, sample_water, sample_dwi, sample_cliniche, target


def main_TRAIN(fold,continue_learning, tipoNormalizzazione_str, loss,
        basePath, classes, ch, learningRate, weightDecay, batchSize, num_epoch,
        vali_set,test_set, esclusi, minDimLesion,
        outputPath, weight_dce, weight_water, weight_dwi):
  print('---------------------------------> Loading data')
  include_train_patient = lambda path: ((path.split('/')[-1].split('_')[0] not in vali_set + test_set) and
                                       (int(path.split('/')[-1].split('_')[-1].split('.')[0])>minDimLesion) and
                                      (path.split('/')[-1].split('_')[0] not in esclusi))

  include_val_patient =  lambda path: ((path.split('/')[-1].split('_')[0] in vali_set) and
                                       (int(path.split('/')[-1].split('_')[-1].split('.')[0])>minDimLesion) and
                                       (path.split('/')[-1].split('_')[0] not in esclusi))

  include_test_patient =  lambda path: ((path.split('/')[-1].split('_')[0] in test_set) and
                                       (int(path.split('/')[-1].split('_')[-1].split('.')[0])>minDimLesion) and
                                       (path.split('/')[-1].split('_')[0] not in esclusi))

  train_files = getFilesForSubset(basePath, classes, include_train_patient)
  print(len(train_files))

  val_files = getFilesForSubset(basePath, classes, include_val_patient)
  print(len(val_files))

  test_files = getFilesForSubset(basePath, classes, include_test_patient)
  print(len(test_files))

  print(' - - - - - train')
  mean_dce_a_train, std_dce_a_train, mean_wat_a_train, std_wat_a_train, mean_dwi_a_train, 
  std_dwi_a_train, mean_dce_c_train, std_dce_c_train, mean_wat_c_train, 
  std_wat_c_train, mean_dwi_c_train, std_dwi_c_train  = statistiche(train_files)

  print(' - - - - - val')
  statistiche(val_files)

  print(' - - - - - test')
  statistiche(test_files)

  if tipoNormalizzazione_str == 'torch_computeMeanAndStd_channnel':
    train_transform_vett, None_transform_vett = definisciTransf(mean_dce_c_train, std_dce_c_train, mean_wat_c_train, std_wat_c_train, mean_dwi_c_train, std_dwi_c_train, tipoNormalizzazione_str)
  else:
   train_transform_vett, None_transform_vett = definisciTransf(mean_dce_a_train, std_dce_a_train, mean_wat_a_train, std_wat_a_train, mean_dwi_a_train, std_dwi_a_train, tipoNormalizzazione_str)

  print(train_transform_vett)
  print(None_transform_vett)


  Trainset = []
  for c in classes:
    print(' Loading ' + c)
    is_valid_class = lambda path: c == path.split('/')[1]
    check_file = lambda path: include_train_patient(path) and is_valid_class(path)
    Trainset.append(My_DatasetFolder(root = basePath, transform= train_transform_vett, is_valid_file=check_file, list_classes=classes ))


  print('0_cavo elements ', str(len(Trainset[0].samples)))
  print('1_cavo elements ', str(len(Trainset[1].samples)))

  num_1 = len(Trainset[1].samples)
  num_0 = len(Trainset[0].samples)

  completeTrainSet = BalanceConcatDataset(Trainset)
  print('0_cavo elements ', str(len(completeTrainSet.datasets[0].samples)))
  print('1_cavo elements ', str(len(completeTrainSet.datasets[1].samples)))

  Val  = My_DatasetFolder(root = basePath, transform=None_transform_vett, is_valid_file=include_val_patient, list_classes=classes)
  Test = My_DatasetFolder(root = basePath, transform=None_transform_vett, is_valid_file=include_test_patient, list_classes=classes)
  print('Validation ',str(len(Val.samples)))
  print('Test ', str(len(Test.samples)))

  print('---------------------------------> TRAINING')

  model_conv = My3DNet_combined(4,1,1,ch,False, weight_dce,weight_water, weight_dwi) #change the model file in the case of resnet
  model_conv = model_conv.cuda()
  print(count_parameters(model_conv))

  optimizer_conv = optim.Adam(model_conv.parameters(), lr=learningRate, weight_decay= weightDecay)
  if loss == 'focal':
    criterionCNN = FocalLoss()
  else:
    criterionCNN = nn.CrossEntropyLoss()

  loader_opts = {'batch_size': batchSize, 'num_workers': 0, 'pin_memory': False}
  print('     Before Training: GPU Memory  %d bytes'%(torch.cuda.memory_allocated()))

  if not continue_learning:
    #inizializzazione senza check point
    best_acc = 0.0
    best_loss = 0.0
    best_epoca = 0
    startEpoch = 1
  else:
    print('RELOAD')
    stato = sio.loadmat(outputPath + 'check_point.mat')
    best_acc = stato['best_acc'][0][0]
    best_loss = stato['best_loss'][0][0]
    best_epoca = stato['best_epoca'][0][0]
    startEpoch = stato['last_epoch'][0][0] + 1
    model_conv.load_state_dict(torch.load(outputPath + 'train_weights.pth'))

  model_conv = train_loop_validation(model_conv,
                                     completeTrainSet, Val, Test,
                                     startEpoch, num_epoch,
                                     loader_opts,
                                     criterionCNN, optimizer_conv,
                                     best_acc, best_loss, best_epoca,
                                     outputPath)

  print('     After Training: GPU Memory  %d bytes'%(torch.cuda.memory_allocated()))
  model_conv.cpu()
  del model_conv
  print('     After Training: GPU Memory  %d bytes'%(torch.cuda.memory_allocated()))

  print('---------------------------------> BEST MODEL')
  lossModel_Train = []
  lossModel_val = []
  lossModel_val_weighted = []

  accModel_Train = []
  accModel_val = []

  Acc_0 = []
  Acc_1 = []

  file = open(outputPath + 'lossTrain.txt', 'r')
  Testo = file.readlines()
  for element in Testo:
    lossModel_Train.append(float(element))

  file = open(outputPath + 'lossVal.txt', 'r')
  Testo = file.readlines()
  for element in Testo:
    lossModel_val.append(float(element))

  plt.figure()
  plt.title("Model: Training Vs Validation Losses")
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.plot(list(range(1,len(lossModel_Train)+1)), lossModel_Train, color='r', label="Training Loss")
  plt.plot(list(range(1, len(lossModel_val)+1)), lossModel_val, color='g', label="Validation Loss")
  plt.legend()
  plt.savefig(outputPath + 'LossTrainVal.png')


  file = open(outputPath + 'AccTrain.txt', 'r')
  Testo = file.readlines()
  for element in Testo:
    accModel_Train.append(float(element))

  file = open(outputPath + 'AccVal.txt', 'r')
  Testo = file.readlines()
  for element in Testo:
    accModel_val.append(float(element))

  plt.figure()
  plt.title("Training Vs Validation Accuracies")
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.plot(list(range(1, len(accModel_Train)+1)), accModel_Train, color='r', label="Training Accuracy")
  plt.plot(list(range(1, len(accModel_val)+1)), accModel_val, color='g', label="Validation Accuracy")
  plt.legend()
  plt.savefig(outputPath + 'AccTrainVal.png')


  file = open(outputPath + 'AccVal_0.txt', 'r')
  Testo = file.readlines()
  for element in Testo:
    Acc_0.append(float(element))

  file = open(outputPath + 'AccVal_1.txt', 'r')
  Testo = file.readlines()
  for element in Testo:
    Acc_1.append(float(element))

  plt.figure()
  plt.title("Validation Accuracies")
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.plot(list(range(1, len(Acc_0)+1)), Acc_0, color='r', label="Val Accuracy class 0")
  plt.plot(list(range(1, len(Acc_1)+1)), Acc_1, color='g', label="Val Accuracy class 1")
  plt.plot(list(range(1, len(accModel_val)+1)), accModel_val, color='b', label="Total Val Accuracy")
  plt.legend()
  plt.savefig(outputPath + 'AccVal.png')

  #PERFORMANCE WITH THE BEST-WEIGHTS
  model_conv = My3DNet_combined(4,1,1,ch,False, weight_dce,weight_water, weight_dwi)
  print(count_parameters(model_conv))
  model_conv.load_state_dict(torch.load(outputPath + 'best_model_weights.pth'))
  model_conv = model_conv.cuda()

  model_conv.eval()
  #model_conv, test, transform
  tabella = prediction_on_Test(model_conv, Test, None_transform_vett)
  tabella.to_csv(outputPath + 'TabellaBestModel.csv', sep = ',', index=False)


  accuracy = np.sum(tabella.true_class.values == tabella.predicted.values)/tabella.shape[0]
  t0 = tabella[tabella.true_class == 0]
  t1 = tabella[tabella.true_class == 1]

  accuracy_0 = np.sum(t0.true_class.values == t0.predicted.values)/t0.shape[0]
  accuracy_1 = np.sum(t1.true_class.values == t1.predicted.values)/t1.shape[0]

  print('Accuracy')
  print(accuracy)
  print('Accuracy_0')
  print(accuracy_0)
  print('Accuracy_1')
  print(accuracy_1)

  model_conv.cpu()
  del model_conv
  print('     After Training: GPU Memory  %d bytes'%(torch.cuda.memory_allocated()))

#Merge training and validation set to performe the final training step
def main_final_restrain(fold,continue_learning_retrain, tipoNormalizzazione_str, loss,
     basePath, classes, ch, learningRate, weightDecay, batchSize,
     vali_set, test_set, esclusi, minDimLesion,
     outputPath, weight_dce, weight_water, weight_dwi):

  include_train_patient_fine = lambda path: ((path.split('/')[-1].split('_')[0] not in test_set) and
                                        (int(path.split('/')[-1].split('_')[-1].split('.')[0])>minDimLesion) and
                                        (path.split('/')[-1].split('_')[0] not in esclusi))

  include_test_patient =  lambda path: ((path.split('/')[-1].split('_')[0] in test_set) and
                                         (int(path.split('/')[-1].split('_')[-1].split('.')[0])>minDimLesion) and
                                         (path.split('/')[-1].split('_')[0] not in esclusi))

  train_files = getFilesForSubset(basePath, classes, include_train_patient_fine)
  print(len(train_files))

  print(' - - - - - train')
  mean_dce_a_train, std_dce_a_train, 
  mean_wat_a_train, std_wat_a_train, 
  mean_dwi_a_train, std_dwi_a_train, 
  mean_dce_c_train, std_dce_c_train,
  mean_wat_c_train, std_wat_c_train, 
  mean_dwi_c_train, std_dwi_c_train  = statistiche(train_files)


  if tipoNormalizzazione_str == 'torch_computeMeanAndStd_channnel':
    train_transform_vett, None_transform_vett = definisciTransf(mean_dce_c_train, std_dce_c_train, mean_wat_c_train, std_wat_c_train, mean_dwi_c_train, std_dwi_c_train, tipoNormalizzazione_str)
  else:
   train_transform_vett, None_transform_vett = definisciTransf(mean_dce_a_train, std_dce_a_train, mean_wat_a_train, std_wat_a_train, mean_dwi_a_train, std_dwi_a_train, tipoNormalizzazione_str)

  print(train_transform_vett)
  print(None_transform_vett)
  #----------------------------------------------------> fine tuning finale

  Trainset = []
  for c in classes:
    print(' Loading ' + c)
    is_valid_class = lambda path: c == path.split('/')[1]
    check_file = lambda path: include_train_patient_fine(path) and is_valid_class(path)
    Trainset.append(My_DatasetFolder(root = basePath, transform=train_transform_vett, is_valid_file=check_file, list_classes=classes))
  print('0_cavo elements ', str(len(Trainset[0].samples)))
  print('1_cavo elements ', str(len(Trainset[1].samples)))
  completeTrainSet = BalanceConcatDataset(Trainset)
  print('0_cavo elements ', str(len(completeTrainSet.datasets[0].samples)))
  print('1_cavo elements ', str(len(completeTrainSet.datasets[1].samples)))

  Test = My_DatasetFolder(root = basePath, transform=None_transform_vett, is_valid_file=include_test_patient, list_classes=classes)
  print('Test ', str(len(Test.samples)))

  #-------------------------------------------------------------------------------> Definizione del modello
  #definizione del modello
  model_conv = My3DNet_combined(4,1,1,ch,False, weight_dce,weight_water, weight_dwi)
  model_conv = model_conv.cuda()
  print(count_parameters(model_conv))

  optimizer_conv = optim.Adam(model_conv.parameters(), lr=learningRate, weight_decay= weightDecay)
  if loss == 'focal':
    criterionCNN = FocalLoss()
  else:
    criterionCNN = nn.CrossEntropyLoss()
  loader_opts = {'batch_size': batchSize, 'num_workers': 0, 'pin_memory': False}

  print('     Before Training: GPU Memory  %d bytes'%(torch.cuda.memory_allocated()))
  stato = sio.loadmat(outputPath + 'check_point.mat')
  best_epoca_onMinLoss = stato['best_epoca'][0][0]

  accModel_val = []
  file = open(outputPath + 'AccVal.txt', 'r')
  Testo = file.readlines()
  for element in Testo:
    accModel_val.append(float(element))

  best_epoca_onMax = np.argmax(accModel_val) +1

  print(best_epoca_onMax)
  best_epoca = np.max([best_epoca_onMinLoss, best_epoca_onMax])

  print('Retrain for ' + str(best_epoca) + ' epochs')
  startEpoch = 1

  if continue_learning_retrain:
    print('RELOAD')
    stato = sio.loadmat(outputPath + 'check_point_for_retrain.mat')
    startEpoch = stato['last_epoch'][0][0] + 1
    model_conv.load_state_dict(torch.load(outputPath + 'FinalWeights.pth'))

  #------------------------------------------------------------ training
  model_conv = train_loop(model_conv,
                        completeTrainSet, Test,
                        startEpoch,
                        best_epoca, loader_opts,
                        criterionCNN, optimizer_conv, outputPath, 'FinalWeights.pth', 'check_point_for_retrain.mat')

  model_conv.cpu()
  del model_conv
  print('     After Training: GPU Memory  %d bytes'%(torch.cuda.memory_allocated()))

  #definizione del modello
  model_conv = My3DNet_combined(4,1,1,ch,False, weight_dce,weight_water, weight_dwi)
  model_conv.load_state_dict(torch.load(outputPath + 'FinalWeights.pth'))
  model_conv = model_conv.cuda()
  print(count_parameters(model_conv))


  model_conv.eval()
  tabella = prediction_on_Test(model_conv, Test, None_transform_vett)
  tabella.to_csv(outputPath + 'TabellaFinalModel.csv', sep = ',', index=False)


  accuracy = np.sum(tabella.true_class.values == tabella.predicted.values)/tabella.shape[0]
  t0 = tabella[tabella.true_class == 0]
  t1 = tabella[tabella.true_class == 1]

  accuracy_0 = np.sum(t0.true_class.values == t0.predicted.values)/t0.shape[0]
  accuracy_1 = np.sum(t1.true_class.values == t1.predicted.values)/t1.shape[0]

  print('Accuracy')
  print(accuracy)
  print('Accuracy_0')
  print(accuracy_0)
  print('Accuracy_1')
  print(accuracy_1)

  model_conv.cpu()
  del model_conv
  print('     After Training: GPU Memory  %d bytes'%(torch.cuda.memory_allocated()))
  
