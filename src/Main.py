from Utils import*
from Model_resnet import* 
#from Model import* (in the case of BasicNet)


tipofunzione = 'train'
fold = 0  #fold for the CV
continue_learning = 'false'
tipoNormalizzazione_str = 'None'
loss = 'cross'
model_type = 'resnet34'  #ResNet version
  
if continue_learning == 'true':
  continue_learning = True
else:
  continue_learning = False
  
exclude = ['115-116'] #patient to exclude
dfs = pd.read_excel('CV_FOLD.xlsx', sheet_name='Foglio1')
classes = ['0_cavo', '1_cavo']
basePath = 'Dataset'

pathforDCEWeights = 'DCE_resnet34'
pathforWATERWeights = 'Water_resnet34'
pathforDWIWeights = 'DWI_3resnet34'
fileweights = 'best_model_weights.pth'

minDimLesion = 30
learningRate = 0.000001
weightDecay = 0.0001
batchSize = 32
num_epoch = 500
ch = 8


test_set = dfs[dfs.FOLD == fold].ID.to_list()
valFold = dfs[dfs.FOLD == fold].Fold_to_use.values[0]
vali_set = dfs[dfs.FOLD == valFold].ID.to_list()
print('Test: fold ' + str(fold))
print(test_set)
print('Val: fold ' + str(valFold))
print(vali_set)
  
if len(list(set(test_set) & set(vali_set)))>0:
  print('error!!!')
  
outputPath = model_type+ '/Fold' + str(fold) +'/'

pathforDCEWeights = pathforDCEWeights + '/Fold' + str(fold) +'/' + fileweights
pathforWATERWeights = pathforWATERWeights + '/Fold' + str(fold) +'/' + fileweights
pathforDWIWeights = pathforDWIWeights + '/Fold' + str(fold) +'/' + fileweights

try:
  os.makedirs(outputPath)
except OSError:
  pass
    
  
  
if tipofunzione == 'train':
  main_TRAIN(fold,continue_learning, tipoNormalizzazione_str, loss,
             basePath, classes, ch, learningRate, weightDecay, batchSize, num_epoch, 
             vali_set, test_set, exclude, minDimLesion,
             outputPath, pathforDCEWeights, pathforWATERWeights, pathforDWIWeights, model_type)
  
  main_final_restrain(fold,False, tipoNormalizzazione_str, loss,
                     basePath, classes, ch, learningRate, weightDecay, batchSize, 
                     vali_set, test_set, exclude, minDimLesion,
                     outputPath, pathforDCEWeights, pathforWATERWeights, pathforDWIWeights, model_type)
else:
  main_final_restrain(fold,continue_learning, tipoNormalizzazione_str,  loss,
                     basePath, classes, ch, learningRate, weightDecay, batchSize, 
                     vali_set, test_set, exclude, minDimLesion,
                     outputPath, pathforDCEWeights, pathforWATERWeights, pathforDWIWeights, model_type)  