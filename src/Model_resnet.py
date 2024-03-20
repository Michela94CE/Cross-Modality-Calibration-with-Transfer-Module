from Utils import*
from TransferModule import*

def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        dilation=dilation,
        stride=stride,
        padding=dilation,
        bias=False)


def downsample_basic_block(x, planes, stride, no_cuda=False):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if not no_cuda:
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 sample_input_D,
                 sample_input_H,
                 sample_input_W,
                 num_seg_classes,
                 shortcut_type='B',
                 no_cuda = False):
        self.inplanes = 64
        self.no_cuda = no_cuda
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(
            1,
            64,
            kernel_size=7,
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False)
            
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=2)

        self.conv_seg = nn.Sequential(
                                        nn.ConvTranspose3d(
                                        512 * block.expansion,
                                        32,
                                        2,
                                        stride=2
                                        ),
                                        nn.BatchNorm3d(32),
                                        nn.ReLU(inplace=True),
                                        nn.Conv3d(
                                        32,
                                        32,
                                        kernel_size=3,
                                        stride=(1, 1, 1),
                                        padding=(1, 1, 1),
                                        bias=False), 
                                        nn.BatchNorm3d(32),
                                        nn.ReLU(inplace=True),
                                        nn.Conv3d(
                                        32,
                                        num_seg_classes,
                                        kernel_size=1,
                                        stride=(1, 1, 1),
                                        bias=False) 
                                        )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    no_cuda=self.no_cuda)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv_seg(x)

        return x

def resnet10(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def resnet200(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model
    
    
  
class Identity(nn.Module):
  def __init__(self):
    super(Identity, self).__init__()
  
  def forward(self, x):
    return x

class MyMedicalResNet(nn.Module):
  def __init__(self, typeResNet, weightPath, num_classes, mode):
    super().__init__()
    if typeResNet == 'resnet50':
      self.backBone = resnet50(sample_input_D = 28, sample_input_H = 28, sample_input_W=28, num_seg_classes=num_classes)
    
    elif typeResNet == 'resnet34':
      self.backBone = resnet34(sample_input_D = 28, sample_input_H = 28, sample_input_W=28, num_seg_classes=num_classes)
    
    elif typeResNet == 'resnet10':
      self.backBone = resnet10(sample_input_D = 28, sample_input_H = 28, sample_input_W=28, num_seg_classes=num_classes)
    
    elif typeResNet == 'resnet18':
      self.backBone = resnet18(sample_input_D = 28, sample_input_H = 28, sample_input_W=28, num_seg_classes=num_classes)
      
    if weightPath != '':
      print('Loading from ' + weightPath)
      self.net_dict = self.backBone.state_dict()
      pretrain = torch.load(weightPath)
      pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in self.net_dict.keys()}
      self.net_dict.update(pretrain_dict)
      self.backBone.load_state_dict(self.net_dict)
    
    if mode == 'dce':
      self.backBone.conv1 = nn.Conv3d(in_channels= 4, out_channels= 64, kernel_size= (7,7,7), stride=(2,2,2), padding=(3,3,3), bias=False)
    
    self.backBone.conv_seg = nn.AdaptiveAvgPool3d((1,1,1))
    
    if typeResNet == 'resnet50':
      self.end = nn.Sequential(
          nn.Linear(2048,num_classes)
          )
    else:
      self.end = nn.Sequential(
          nn.Linear(512,num_classes)   
          )

    
  def forward(self, x):
    x = self.backBone(x)
    x = torch.flatten(x, 1)
    x = self.end(x)
    return x


class My3DNet_combined(nn.Module):
  def __init__(self, typeResNet_dce, num_classes ,printB, weights_dec, weights_water, weights_dwi):
    super(My3DNet_combined, self).__init__()
    self.printB = printB
    
    self.dce_net = MyMedicalResNet(typeResNet_dce, '', num_classes, 'dce') 
    self.water_net = MyMedicalResNet(typeResNet_dce, '', num_classes, 'water')
    self.dwi_net = MyMedicalResNet(typeResNet_dce, '', num_classes, 'dwi') 
    
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

    self.dce_net.end = Identity()
    self.water_net.end = Identity()
    self.dwi_net.end = Identity()

    if typeResNet_dce == 'resnet50':
      self.fusionAfter1 = CNNFusion(256,256,256, printB)
      self.fusionAfter2 = CNNFusion(512,512,512, printB)
      self.fusionAfter3 = CNNFusion(1024,1024,1024, printB)
      self.fusionAfter4 = CNNFusion(2048,2048,2048, printB)
      final_ch = 2048
    else:
      self.fusionAfter1 = CNNFusion(64,64,64, printB)
      self.fusionAfter2 = CNNFusion(128,128,128, printB)
      self.fusionAfter3 = CNNFusion(256,256,256, printB)
      self.fusionAfter4 = CNNFusion(512,512, 512, printB)
      final_ch =512

    self.last_cnn = nn.Sequential( 
        ReductionCoreBlock(inputChannel=final_ch*3, outchannel=final_ch*3, ksize=(1,1,1), stride=(1,1,1), pad=(0,0,0)),
        )
    
    self.avg = nn.AdaptiveAvgPool3d((1,1,1))

    self.end = nn.Sequential(
        nn.Linear(final_ch*3,final_ch),
        nn.ReLU(inplace= True),
        nn.Linear(final_ch,num_classes),
        )
    
  def step1(self,dce_net, water_net, dwi_net, dce_v, water_v,dwi_v):
    dce_v = dce_net.conv1(dce_v)
    dce_v = dce_net.bn1(dce_v)
    dce_v = dce_net.relu(dce_v)
    dce_v = dce_net.maxpool(dce_v)
    dce_v = dce_net.layer1(dce_v)
    
    water_v = water_net.conv1(water_v)
    water_v = water_net.bn1(water_v)
    water_v = water_net.relu(water_v)
    water_v = water_net.maxpool(water_v)
    water_v = water_net.layer1(water_v)

    dwi_v = dwi_net.conv1(dwi_v)
    dwi_v = dwi_net.bn1(dwi_v)
    dwi_v = dwi_net.relu(dwi_v)
    dwi_v = dwi_net.maxpool(dwi_v)
    dwi_v = dwi_net.layer1(dwi_v)
    return dce_v, water_v,dwi_v
    
  def step2(self,dce_net, water_net, dwi_net, dce_v, water_v,dwi_v):
   dce_v = dce_net.layer2(dce_v)
   water_v = water_net.layer2(water_v)
   dwi_v = dwi_net.layer2(dwi_v)
   return dce_v, water_v,dwi_v
   
   
  def step3(self,dce_net, water_net, dwi_net, dce_v, water_v,dwi_v):
   dce_v = dce_net.layer3(dce_v)
   water_v = water_net.layer3(water_v)
   dwi_v = dwi_net.layer3(dwi_v)
   return dce_v, water_v,dwi_v
   
  def step4(self,dce_net, water_net, dwi_net, dce_v, water_v,dwi_v):
   dce_v = dce_net.layer4(dce_v)
   water_v = water_net.layer4(water_v)
   dwi_v = dwi_net.layer4(dwi_v)
   return dce_v, water_v,dwi_v

  def forward(self, dce_v, water_v, dwi_v):
    if self.printB:
      print(dce_v.shape)
      print(water_v.shape)
      print(dwi_v.shape)

    #layer 1
    dce_v, water_v, dwi_v =  self.step1(self.dce_net.backBone, self.water_net.backBone, self.dwi_net.backBone, dce_v, water_v,dwi_v)
    dce_v, water_v, dwi_v =self.fusionAfter1(dce_v,water_v, dwi_v)
    if self.printB:
      print('After1')
      print(dce_v.shape)
      print(water_v.shape)
      print(dwi_v.shape)

    #layer 2
    dce_v, water_v, dwi_v =  self.step2(self.dce_net.backBone, self.water_net.backBone, self.dwi_net.backBone, dce_v, water_v,dwi_v)
    dce_v, water_v, dwi_v =self.fusionAfter2(dce_v,water_v, dwi_v)
    if self.printB:
      print('After2')
      print(dce_v.shape)
      print(water_v.shape)
      print(dwi_v.shape)

    #layer 3
    dce_v, water_v, dwi_v =  self.step3(self.dce_net.backBone, self.water_net.backBone, self.dwi_net.backBone, dce_v, water_v,dwi_v)
    dce_v, water_v, dwi_v = self.fusionAfter3(dce_v,water_v, dwi_v)

    if self.printB:
      print('After3')
      print(dce_v.shape)
      print(water_v.shape)
      print(dwi_v.shape)

    #layer 4
    dce_v, water_v, dwi_v =  self.step4(self.dce_net.backBone, self.water_net.backBone, self.dwi_net.backBone, dce_v, water_v,dwi_v)
    dce_v, water_v, dwi_v =self.fusionAfter4(dce_v,water_v, dwi_v)

    if self.printB:
      print('After4')
      print(dce_v.shape)
      print(water_v.shape)
      print(dwi_v.shape)
    
    x_img = torch.cat((dce_v,water_v,dwi_v), 1)
    if self.printB:
      print(x_img.shape)

    x_img = self.last_cnn(x_img)
    if self.printB:
      print(x_img.shape)

    x_img = self.avg(x_img)  
    if self.printB:
      print('dopoAVG')
      print(x_img.shape)
    
    x_img = torch.flatten(x_img,1)
    x = self.end(x_img)
    return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)