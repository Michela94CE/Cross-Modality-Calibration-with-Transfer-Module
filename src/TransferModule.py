from Utils import*

# IMPLEMENTATION OF THE TRANSFER MODULE PRESENTED IN THE PAPER
class CNNFusion(nn.Module):
  def __init__(self, channel_dce, channel_wat, channel_dwi, printB):
    super(CNNFusion, self).__init__()
    self.printB = printB
    self.avg1 = nn.AdaptiveAvgPool3d((1,1,1))
    self.avg2 = nn.AdaptiveAvgPool3d((1,1,1))
    self.avg3 = nn.AdaptiveAvgPool3d((1,1,1))

    ratio = int((channel_dce + channel_wat + channel_dwi )/4)

    self.unique = nn.Sequential(
       nn.Linear(channel_dce + channel_wat + channel_dwi, ratio),
        nn.ReLU(inplace= True),
    )

    self.vector_dce_wat = nn.Sequential(
        #merge dce wat for dwi calibration
        nn.Linear(ratio, channel_dwi),
        nn.Sigmoid()

    )

    self.vector_dce_dwi = nn.Sequential(
        #merge dce dwi for wat calibration
        nn.Linear(ratio, channel_wat ),
        nn.Sigmoid()
    )

    self.vector_wat_dwi = nn.Sequential(
        #merge wat dwi for dce calibration
        nn.Linear(ratio, channel_dce),
        nn.Sigmoid()

    )

  def forward(self, dce, wat, dwi):
    if self.printB:
      print(dce.shape)
      print(wat.shape)
      print(dwi.shape)

    dce_av = torch.flatten(self.avg1(dce),1) 
    wat_av = torch.flatten(self.avg2(wat),1) 
    dwi_av = torch.flatten(self.avg3(dwi),1) 

    if self.printB:
      print('avg pool')
      print(dce_av.shape)
      print(wat_av.shape)
      print(dwi_av.shape)

    unique = self.unique(torch.cat((dce_av, wat_av, dwi_av), 1)) #shared represetation

    if self.printB:
      print('unique')
      print(unique.shape)

    dce_wat = self.vector_dce_wat(unique) 
    dce_dwi = self.vector_dce_dwi(unique) 
    wat_dwi =  self.vector_wat_dwi(unique) 


    if self.printB:
      print(dce_wat.shape)
      print(dce_dwi.shape)
      print(wat_dwi.shape)

    dce_f = dce * wat_dwi.unsqueeze(2).unsqueeze(3).unsqueeze(4)
    wat_f = wat * dce_dwi.unsqueeze(2).unsqueeze(3).unsqueeze(4)
    dwi_f = dwi * dce_wat.unsqueeze(2).unsqueeze(3).unsqueeze(4)

    if self.printB:
      print('ris ')
      print(dce_f.shape)
      print(wat_f.shape)
      print(dwi_f.shape)

    return dce_f, wat_f, dwi_f  #calibrated features maps