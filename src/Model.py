from Utils import*
from TransferModule import*

#-------------------------------------------------------------------------------> BASIC-NET
class ReductionCoreBlock(nn.Module):
  def __init__(self, inputChannel, outchannel, ksize, stride, pad):
    super(ReductionCoreBlock, self).__init__()

    self.downsample = nn.Sequential(
        nn.Conv3d(in_channels=inputChannel, out_channels=outchannel, kernel_size = ksize, stride = stride, padding=pad, bias=False),
        nn.BatchNorm3d(outchannel),
        nn.ReLU(inplace=True),
        )

  def forward(self, x):
    out = self.downsample(x)
    return out

class Identity(nn.Module):
  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x


class My3DNet(nn.Module):
  def __init__(self, nChannel, nfilters,printB):
    super(My3DNet, self).__init__()
    self.printB = printB
    self.reduction_1 =  nn.Sequential(
        ReductionCoreBlock(inputChannel = nChannel, outchannel=nfilters, ksize=(4,4,4), stride=(2,2,2), pad=(1,1,1)),
    )

    self.reduction_2 = nn.Sequential(
        ReductionCoreBlock(inputChannel= nfilters, outchannel=nfilters*2, ksize=(4,4,4), stride=(2,2,2), pad=(1,1,1)),
    )

    self.reduction_3 = nn.Sequential(
        ReductionCoreBlock(inputChannel= nfilters*2, outchannel=nfilters*4, ksize=(4,4,4), stride=(2,2,2), pad=(1,1,1)),
        )

    self.reduction_4 =  nn.Sequential(
        ReductionCoreBlock(inputChannel= nfilters*4, outchannel=nfilters*8, ksize=(4,4,4), stride=(2,2,2), pad=(1,1,1)),
    )

    self.reduction_5 =  nn.Sequential(
        ReductionCoreBlock(inputChannel= nfilters*8, outchannel=nfilters*16, ksize=(4,4,4), stride=(2,2,2), pad=(0,0,0)),
    )

    self.end =  nn.Sequential(
        nn.Linear(nfilters*16,nfilters*8),
        nn.ReLU(inplace= True),
        nn.Linear(nfilters*8,2),
        )


  def forward(self, x):
    if self.printB:
      print(x.shape)
    x = self.reduction_1(x)

    if self.printB:
      print(x.shape)
    x = self.reduction_2(x)

    if self.printB:
      print(x.shape)
    x = self.reduction_3(x)
    if self.printB:
      print(x.shape)

    x = self.reduction_4(x)
    if self.printB:
      print(x.shape)
    x = self.reduction_5(x)
    if self.printB:
      print(x.shape)

    x = torch.flatten(x,1)
    if self.printB:
      print(x.shape)
    x = self.end(x)

    return x

class My3DNet_combined(nn.Module):
  def __init__(self, nChannel_dce, nChannel_water, nChannel_dwi, nfilters,printB, weights_dec, weights_water, weights_dwi):
    super(My3DNet_combined, self).__init__()
    self.printB = printB

    self.dce_net = My3DNet(nChannel_dce,nfilters, False)
    self.water_net = My3DNet(nChannel_water,nfilters, False)
    self.dwi_net = My3DNet(nChannel_dwi,nfilters, False)

    #carico i pesi
    if weights_dec == '':
      print('No DCE inizialization')
    else:
      self.dce_net.load_state_dict(torch.load(weights_dec))

    if weights_water == '':
      print('No WATER inizialization')
    else:
      self.water_net.load_state_dict(torch.load(weights_water))

    if weights_dwi == '':
      print('No DWI inizialization')
    else:
      self.dwi_net.load_state_dict(torch.load(weights_dwi))

    #modifica della rete
    self.dce_net.end = Identity()
    self.water_net.end = Identity()
    self.dwi_net.end = Identity()

    #new layers
    self.clinic_net = nn.Sequential(
        nn.Linear(10,4),
        nn.BatchNorm1d(4),
        nn.ReLU(inplace= True),
    )

    self.fusionAfter3 = CNNFusion(nfilters*4,nfilters*4,nfilters*4, printB)
    self.fusionAfter4 = CNNFusion(nfilters*8,nfilters*8,nfilters*8, printB)
    self.fusionAfter5 = CNNFusion(nfilters*16,nfilters*16,nfilters*16, printB)

    self.last_cnn = nn.Sequential(
        ReductionCoreBlock(inputChannel= nfilters*48, outchannel=nfilters*48, ksize=(1,1,1), stride=(1,1,1), pad=(0,0,0)),
        )


    self.end = nn.Sequential(
        nn.Linear(nfilters*48+4,nfilters*24),
        nn.ReLU(inplace= True),
        nn.Linear(nfilters*24,2),
        )

  def forward(self, dce_v, water_v, dwi_v, feature_clnic):
    if self.printB:
      print(dce_v.shape)
      print(water_v.shape)
      print(dwi_v.shape)

    #layer 1
    dce_v = self.dce_net.reduction_1(dce_v)
    water_v = self.water_net.reduction_1(water_v)
    dwi_v = self.dwi_net.reduction_1(dwi_v)

    if self.printB:
      print(dce_v.shape)
      print(water_v.shape)
      print(dwi_v.shape)

    #layer 2
    dce_v = self.dce_net.reduction_2(dce_v)
    water_v = self.water_net.reduction_2(water_v)
    dwi_v = self.dwi_net.reduction_2(dwi_v)

    if self.printB:
      print(dce_v.shape)
      print(water_v.shape)
      print(dwi_v.shape)

    #layer 3
    dce_v = self.dce_net.reduction_3(dce_v)
    water_v = self.water_net.reduction_3(water_v)
    dwi_v = self.dwi_net.reduction_3(dwi_v)
    dce_v, water_v, dwi_v = self.fusionAfter3(dce_v,water_v, dwi_v)

    if self.printB:
      print(dce_v.shape)
      print(water_v.shape)
      print(dwi_v.shape)

    #layer 4
    dce_v = self.dce_net.reduction_4(dce_v)
    water_v = self.water_net.reduction_4(water_v)
    dwi_v = self.dwi_net.reduction_4(dwi_v)
    dce_v, water_v, dwi_v =self.fusionAfter4(dce_v,water_v, dwi_v)

    if self.printB:
      print(dce_v.shape)
      print(water_v.shape)
      print(dwi_v.shape)

    #layer 5
    dce_v = self.dce_net.reduction_5(dce_v)
    water_v = self.water_net.reduction_5(water_v)
    dwi_v = self.dwi_net.reduction_5(dwi_v)
    dce_v, water_v, dwi_v =self.fusionAfter5(dce_v,water_v, dwi_v)


    if self.printB:
      print(dce_v.shape)
      print(water_v.shape)
      print(dwi_v.shape)

    #----> tabular Features
    if self.printB:
      print(feature_clnic.shape)
    feature_clnic = self.clinic_net(feature_clnic)
    if self.printB:
      print(feature_clnic.shape)

    x_img = torch.cat((dce_v,water_v,dwi_v), 1)
    if self.printB:
      print(x_img.shape)
    x_img = self.last_cnn(x_img)
    if self.printB:
      print(x_img.shape)

    x_img = torch.flatten(x_img,1)
    x = torch.cat((x_img, feature_clnic), 1)
    if self.printB:
      print(x.shape)
    x = self.end(x)

    return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)