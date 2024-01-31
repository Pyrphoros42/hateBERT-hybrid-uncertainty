from manual_training_inference import *
#from manual_training_inference_unmodified import *
from Preprocess.dataCollect import *
from Models.utils import load_model
import warnings
warnings.filterwarnings('ignore')
#torch.backends.cudnn.benchmark = True
import argparse


#path_file='Saved/bert-base-uncased_11_6_3_0.001/config.json'
path_file='best_model_json/bestModel_bert_base_uncased_Attn_train_TRUE.json'
with open(path_file,mode='r') as f:
    params = json.load(f)
for key in params:
    if params[key] == 'True':
          params[key]=True
    elif params[key] == 'False':
          params[key]=False
    if( key in ['batch_size','num_classes','hidden_size','supervised_layer_pos','num_supervised_heads','random_seed','max_length']):
        if(params[key]!='N/A'):
            params[key]=int(params[key])

    if((key == 'weights') and (params['auto_weights']==False)):
        params[key] = ast.literal_eval(params[key])


##### change in logging to output the results to neptune
params['logging']='local'
params['device']='cuda'
params['best_params']=False

if torch.cuda.is_available() and params['device']=='cuda':
    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")
else:
    print('Since you dont want to use GPU, using the CPU instead.')
    device = torch.device("cpu")

dict_data_folder={
      '2':{'data_file':'Data/dataset.json','class_label':'Data/classes_two.npy'},
      '3':{'data_file':'Data/dataset.json','class_label':'Data/classes.npy'}
}

#### Few handy keys that you can directly change.
params['num_dropouts']=3
params['variance']=1
params['dropout_bert']=0.1
params['epochs']=3
params['to_save']=True
params['num_classes']=3
params['AU']=0
params['EU']=1
assert params['AU'] == 0 or params['EU'] == 0
if params['EU']:
    params['path_files']='Saved/EU_without_Hispanic_Refugee_bert-base-uncased_11_6_4_0.001'
params['data_file']=dict_data_folder[str(params['num_classes'])]['data_file']
params['class_names']=dict_data_folder[str(params['num_classes'])]['class_label']
if(params['num_classes']==2 and (params['auto_weights']==False)):
      params['weights']=[1.0,1.0]

#for att_lambda in [0.001,0.01,0.1,1,10,100]

train_bool = 0
get_AU_data = 0
get_EU_data = False

if train_bool:
    if get_AU_data:
        data = get_annotated_data(params)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=False)
        get_AU_training_data(data, params, tokenizer)
    elif get_EU_data:
        data = get_annotated_data(params)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=False)
        get_EU_training_data(data, params, tokenizer)
    else:
        train_model(params,device)


else:
    model=select_model(params,embeddings=None)
    train,val,test=createDatasetSplit(params)
    test_dataloader=combine_features(test,params,is_train=False)
    Eval_phase(params, device, model=model, test_dataloader=test_dataloader)
