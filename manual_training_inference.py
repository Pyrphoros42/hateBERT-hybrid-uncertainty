### This is run when you want to select the parameters from the parameters file
import transformers 
import torch
import neptune
from knockknock import slack_sender
from transformers import *
import glob
from scipy.special import softmax
import random
import pandas as pd
from transformers import BertTokenizer

from HateXplain.plotFigures import plot_correlation_AU_TU
from Models.utils import masked_cross_entropy,fix_the_random,format_time,save_normal_model,save_bert_model,\
    predictive_entropy, expected_entropy, mutual_info, softmax_old
from tqdm import tqdm
from TensorDataset.datsetSplitter import createDatasetSplit
from TensorDataset.dataLoader import combine_features
from Preprocess.dataCollect import collect_data,set_name
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,recall_score,precision_score
import matplotlib.pyplot as plt
import time
import os
import GPUtil
from sklearn.utils import class_weight
import json
from Models.bertModels import *
from Models.otherModels import *
from plotFigures import *
import sys
import time
from waiting import wait
from sklearn.preprocessing import LabelEncoder, RobustScaler
import numpy as np
import threading
import argparse
import ast

# https://softwareengineering.stackexchange.com/questions/254279/why-doesnt-python-have-a-flatten-function-for-lists
def flatten(l): return flatten(l[0]) + (flatten(l[1:]) if len(l) > 1 else []) if type(l) is list else [l]

#def softmax(x):
 #   """Compute softmax values for each sets of scores in x."""
  #  e_x = np.exp(x - np.max(x))
   # return e_x / e_x.sum(axis=0) # only difference

### gpu selection algo
def get_gpu():
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    while(1):
        tempID = [] 
        tempID = GPUtil.getAvailable(order = 'memory', limit = 1, maxLoad = 0.1, maxMemory = 0.07, includeNan=False, excludeID=[], excludeUUID=[])
        if len(tempID) > 0:
            print("Found a gpu")
            print('We will use the GPU:',tempID[0],torch.cuda.get_device_name(tempID[0]))
            deviceID=tempID
            return deviceID
        else:
            time.sleep(5)
#    return flag,deviceID


##### selects the type of model
def select_model(params,embeddings):
    if(params['bert_tokens']):
        if(params['what_bert']=='weighted'):
            model = SC_weighted_BERT.from_pretrained(
            params['path_files'], # Use the 12-layer BERT model, with an uncased vocab.
            num_labels = params['num_classes'], # The number of output labels
            output_attentions = True, # Whether the model returns attentions weights.
            output_hidden_states = False, # Whether the model returns all hidden-states.
            hidden_dropout_prob=params['dropout_bert'],
            params=params
            )
            print('Running base BERT')
        else:
            print("Error in bert model name!!!!")
        return model
    else:
        text=params['model_name']
        if(text=="birnn"):
            model=BiRNN(params,embeddings)
        elif(text == "birnnatt"):
            model=BiAtt_RNN(params,embeddings,return_att=False,)
        elif(text == "birnnscrat"):
            model=BiAtt_RNN(params,embeddings,return_att=True)
        elif(text == "cnn_gru"):
            model=CNN_GRU(params,embeddings)
        elif(text == "lstm_bad"):
            model=LSTM_bad(params)
        else:
            print("Error in model name!!!!")
        return model

# activating dropout during eval
# https://stackoverflow.com/questions/66828446/make-predictions-on-on-huggingfaces-bert-with-dropout-on
def apply_dropout(model):
    if type(model) == nn.Dropout:
        model.train()

def remove_indeces(array1,array2,indeces_array):
    a1 = []
    a2 = []
    for i in range(len(array1)):
        if i not in indeces_array:
            a1.append(array1[i])
            a2.append(array2[i])
    return a1, a2

# Returns an array of indeces of the biggest array elements
def get_indeces(array, percentage):
    if not 0 <= percentage < 100:
        print("Percentage should be between 0 and 100%! Given as an integer")
        return 0
    indeces_len = int(len(array)/100*percentage)
    indeces = np.argpartition(array,-indeces_len)[-indeces_len:]
    return indeces


def get_indeces_acc_eval_true_false(pred_labels=0,true_labels=0,indeces=0, eval_array=0):
    true_count = 0
    false_count = 0
    eval_true = 0
    eval_false = 0
    for i in indeces:
        if pred_labels[i] != true_labels[i]:
            false_count+=1
            eval_false += eval_array[i]
        else:
            true_count+=1
            eval_true += eval_array[i]

    accuracy = true_count / (true_count+false_count)
    return_true = return_false = -1
    if true_count:
        return_true = eval_true/true_count
    if false_count:
        return_false = eval_false/false_count

    return accuracy, return_true, return_false


# Return three arrays of each percentage step along the eval_array, with average accuracy, average evaluation of true predictions, average evaluation of false predictions
def percentage_eval(eval_array, pred_labels=0, true_labels=0):
    sorted_eval_array = np.argsort(eval_array)
    average_acc_array = np.zeros(100)
    eval_true_array = np.zeros(100)
    eval_false_array = np.zeros(100)
    start_index = 0
    last_index = 0
    for percentage in range(100):
        start_index = last_index
        last_index = int(len(eval_array)/100*(percentage+1))
        average_acc_array[percentage], eval_true_array[percentage], eval_false_array[percentage] = \
            get_indeces_acc_eval_true_false(pred_labels=pred_labels,true_labels=true_labels, indeces=sorted_eval_array[start_index:last_index], eval_array=eval_array)

    return average_acc_array, eval_true_array, eval_false_array



def Eval_phase(params,which_files='test',model=None,test_dataloader=None,device=None):
    if(params['is_model']==True):
        print("model previously passed")
        if (params['num_dropouts']):
            model.eval()
            model.apply(apply_dropout)

    else:
        return 1
#         ### Have to modify in the final run
#         model=select_model(params['what_bert'],params['path_files'],params['weights'])
#         model.cuda()
#         model.eval()


    print("Running eval on ",which_files,"...")
    t0 = time.time()

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    # Tracking variables 
    
    true_labels=[]
    pred_labels=[]
    logits_all=[]
    probabilities=[]
    dissent=[]
    target=[]
    # Evaluate data for one epoch
    for step, batch in tqdm(enumerate(test_dataloader)):

        # Progress update every 40 batches.
        if step % 40 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)

        # `batch` contains three pytorch tensors:
        #   [0]: input ids 
        #   [1]: attention vals
        #   [2]: attention mask
        #   [3]: labels 
        b_input_ids = batch[0].to(device)
        b_att_val = batch[1].to(device)
        b_input_mask = batch[2].to(device)
        b_labels = batch[3].to(device)
        if params['AU']:
            # 0 = unanimous, 1 = dissent, 2 = undecided
            dissent.extend(batch[4].detach())
            #print(dissent)
        elif params['EU']:
            target.extend(batch[4].detach())


        # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
        #model.zero_grad()
        for weight in model.parameters():
            weight.grad = None
        num_dropouts = params['num_dropouts']
        outputs = []
        #outputs = np.zeros(shape=(num_dropouts, 300)) #, dtype=torch.tensor(requires_grad=False))
        #outputs = torch.tensor([1,2,3], requires_grad=False)
        for i in range(num_dropouts):
            outputs.append(model(b_input_ids,
                attention_vals=b_att_val,
                attention_mask=b_input_mask,
                labels=None,device=device))

        logits = [outputs[i][0] for i in range(num_dropouts)]
        # Move logits and labels to CPU
        logits = [logits[i].detach().cpu().numpy() for i in range(num_dropouts)]
        label_ids = b_labels.detach().cpu().numpy()


        # Calculate the accuracy for this batch of test sentences.
        # Accumulate the total accuracy.
        logits_pred = np.average([logits[i] for i in range(num_dropouts)], axis=0)
        #print(logits_pred)
        pred_labels+=list(np.argmax(logits_pred,axis=1).flatten())
        #print(pred_labels)
        true_labels+=list(label_ids.flatten())
        #print(true_labels)

        new_logits_pred = [list(softmax(logits, axis=-1)[i]) for i in range(len(logits))]
        probabilities.extend(np.moveaxis(new_logits_pred,0,-1))


        # Weirder array of array
        #probabilities.append(softmax(logits_pred))

    # Aleatoric Uncertainty
    exp_entropy = expected_entropy(probabilities)
    # Total Uncertainty
    pred_entropy = predictive_entropy(probabilities)
    # Epistemic Uncertainty = TU - AU
    epis_uncertainty = mutual_info(pred_entropy=pred_entropy, exp_entropy=exp_entropy)


    matcharray = np.zeros(len(true_labels))
    for i in range(len(true_labels)):
        if true_labels[i] == pred_labels[i]:
            matcharray[i] = 1

    # Of those with true label = hate speech, how many of them are there with target = 1?
    # The of the highest X% epistemic uncertainty with true label = hate speech,
    # how many of them are target = 1?
    # Do we need true label = hate speech?
    if params['EU']:
        # is EU uniquely correlated with target groups?
        plot_percentage_target_EU(epis_uncertainty, target, params)
        plot_percentage_target_AU(exp_entropy, target, params)
        plot_percentage_target_TU(pred_entropy, target, params)
        plot_EU_by_target(epis_uncertainty, matcharray, true_labels, target, params)
        plot_AU_by_target(exp_entropy, matcharray, true_labels, target, params)
        plot_TU_by_target(pred_entropy, matcharray, true_labels, target, params)


    if params['AU']:
        # is AU uniquely correlated with number of dissenting voices?
        plot_dissent_TU(pred_entropy, dissent, matcharray, params)
        plot_dissent_EU(epis_uncertainty, dissent, matcharray, params)
        plot_dissent_AU(exp_entropy, dissent, matcharray, params)
        # remove 'undecided' labels for accuracy calculation
        clean_true_labels = []
        clean_pred_labels = []
        clean_pred_entropy = []
        clean_epis_uncertainty= []
        clean_exp_entropy = []
        clean_matcharray = []
        for i in range(len(true_labels)):
            if dissent[i] != 2:
                clean_true_labels.append(true_labels[i])
                clean_pred_labels.append(pred_labels[i])
                clean_pred_entropy.append(pred_entropy[i])
                clean_matcharray.append(matcharray[i])
                clean_epis_uncertainty.append(epis_uncertainty[i])
                clean_exp_entropy.append(exp_entropy[i])

        testf1=f1_score(clean_true_labels, clean_pred_labels, average='macro')
        testacc=accuracy_score(clean_true_labels,clean_pred_labels)
        testprecision=precision_score(clean_true_labels, clean_pred_labels, average='macro')
        testrecall=recall_score(clean_true_labels, clean_pred_labels, average='macro')

        plot_diff_TU(clean_pred_entropy,clean_true_labels, clean_pred_labels, testacc, testf1, testprecision, testrecall, params)

        aleatoric_acc, aleatoric_true, aleatoric_false = percentage_eval(clean_exp_entropy,pred_labels=clean_pred_labels, true_labels=clean_true_labels)
        total_acc, total_true, total_false = percentage_eval(clean_pred_entropy,pred_labels=clean_pred_labels, true_labels=clean_true_labels)
        epistemic_acc, epistemic_true, epistemic_false = percentage_eval(clean_epis_uncertainty,pred_labels=clean_pred_labels, true_labels=clean_true_labels)

        # Plotting AU
        plot_percentage_AU(aleatoric_acc, aleatoric_true, aleatoric_false, exp_entropy, params)



        # Plotting TU
        plot_percentage_TU_total(total_acc, pred_entropy,params)
        plot_correlation_AU_TU(clean_exp_entropy, clean_pred_entropy, params)
        plot_percentage_EU(epistemic_acc, epis_uncertainty, params)

    else:
        testf1=f1_score(true_labels, pred_labels, average='macro')
        testacc=accuracy_score(true_labels,pred_labels)
        testprecision=precision_score(true_labels, pred_labels, average='macro')
        testrecall=recall_score(true_labels, pred_labels, average='macro')
        plot_diff_TU(pred_entropy,true_labels,pred_labels, testacc, testf1, testprecision, testrecall, params)


        aleatoric_acc, aleatoric_true, aleatoric_false = percentage_eval(exp_entropy,pred_labels=pred_labels, true_labels=true_labels)
        total_acc, total_true, total_false = percentage_eval(pred_entropy,pred_labels=pred_labels, true_labels=true_labels)
        epistemic_acc, epistemic_true, epistemic_false = percentage_eval(epis_uncertainty,pred_labels=pred_labels, true_labels=true_labels)

        # Plotting AU
        plot_percentage_AU(aleatoric_acc, aleatoric_true, aleatoric_false, exp_entropy, params)
        plot_correlation_AU_TU(exp_entropy, pred_entropy, params)


        # Plotting TU
        plot_percentage_TU_total(total_acc, pred_entropy,params)

        plot_percentage_EU(epistemic_acc, epis_uncertainty, params)

        plot_highest_EU_TU_AU(epis_uncertainty,pred_entropy,exp_entropy, params)

    if(params['num_classes']==3):
        #new_prob =
        #testrocauc=roc_auc_score(true_labels, pred_labels,multi_class='ovo',average='macro')
        testrocauc=0
    else:
        #testrocauc=roc_auc_score(true_labels, logits_all_final,multi_class='ovo',average='macro')
        testrocauc=0



    if(params['logging']!='neptune' or params['is_model'] == True):
        # Report the final accuracy for this validation run.
        print(" Accuracy: {0:.2f}".format(testacc))
        print(" Fscore: {0:.2f}".format(testf1))
        print(" Precision: {0:.2f}".format(testprecision))
        print(" Recall: {0:.2f}".format(testrecall))
        #print(" Roc Auc: {0:.2f}".format(testrocauc))

        print(" Test took: {:}".format(format_time(time.time() - t0)))
        #print(ConfusionMatrix(true_labels,pred_labels))
        #print()
    else:
        bert_model = params['path_files']
        language  = params['language']
        name_one=bert_model+"_"+language
        neptune.create_experiment(name_one,params=params,send_hardware_metrics=False,run_monitoring_thread=False)
        neptune.append_tag(bert_model)
        neptune.append_tag(language)
        neptune.append_tag('test')
        neptune.log_metric('test_f1score',testf1)
        neptune.log_metric('test_accuracy',testacc)
        neptune.log_metric('test_precision',testprecision)
        neptune.log_metric('test_recall',testrecall)
        neptune.log_metric('test_rocauc',testrocauc)
        neptune.stop()





    return testf1,testacc,testprecision,testrecall#,testrocauc,probabilities

    
    
def train_model(params,device):
    embeddings=None
    if(params['bert_tokens']):
        train,val,test=createDatasetSplit(params)
    else:
        train,val,test,vocab_own=createDatasetSplit(params)
        params['embed_size']=vocab_own.embeddings.shape[1]
        params['vocab_size']=vocab_own.embeddings.shape[0]
        embeddings=vocab_own.embeddings
    if(params['auto_weights']):
        y_test = [ele[2] for ele in test] 
#         print(y_test)
        encoder = LabelEncoder()
        encoder.classes_ = np.load(params['class_names'],allow_pickle=True)
        params['weights']=class_weight.compute_class_weight('balanced',classes=np.unique(y_test),y=y_test).astype('float32')
        #params['weights']=np.array([len(y_test)/y_test.count(encoder.classes_[0]),len(y_test)/y_test.count(encoder.classes_[1]),len(y_test)/y_test.count(encoder.classes_[2])]).astype('float32') 
        

    print(params['weights'])
    train_dataloader =combine_features(train,params,is_train=True)   
    validation_dataloader=combine_features(val,params,is_train=False)
    test_dataloader=combine_features(test,params,is_train=False)
    
   
    model=select_model(params,embeddings)

    if(params["device"]=='cuda'):
        model.cuda()
    optimizer = AdamW(model.parameters(),
                  lr = params['learning_rate'], # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = params['epsilon'] # args.adam_epsilon  - default is 1e-8.
                )


    # Number of training epochs (authors recommend between 2 and 4)
    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * params['epochs']

    # Create the learning rate scheduler.
    if(params['bert_tokens']):
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = int(total_steps/10),                     num_training_steps = total_steps)

    # Set the seed value all over the place to make this reproducible.
    fix_the_random(seed_val = params['random_seed'])
    # Store the average loss after each epoch so we can plot them.
    loss_values = []
    if(params['bert_tokens']):
        bert_model = params['path_files']
        name_one=bert_model
    else:
        name_one=params['model_name']
        
    if(params['logging']=='neptune'):
        neptune.create_experiment(name_one,params=params,send_hardware_metrics=False,run_monitoring_thread=False)
        
        neptune.append_tag(name_one)
        if(params['best_params']):
            neptune.append_tag('AAAI final best')
        else:
            neptune.append_tag('AAAI final')
        
    best_val_fscore=0
    best_test_fscore=0

    best_val_roc_auc=0
    best_test_roc_auc=0
    
    best_val_precision=0
    best_test_precision=0
    
    best_val_recall=0
    best_test_recall=0
    
    
    for epoch_i in range(0, params['epochs']):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, params['epochs']))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_loss = 0
        model.train()

        # For each batch of training data...
        for step, batch in tqdm(enumerate(train_dataloader)):

            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
                
                
            # `batch` contains three pytorch tensors:
            #   [0]: input ids 
            #   [1]: attention vals
            #   [2]: attention mask
            #   [3]: labels 
            b_input_ids = batch[0].to(device)
            b_att_val = batch[1].to(device)
            b_input_mask = batch[2].to(device)
            b_labels = batch[3].to(device)
            if params['EU']:
                #print(batch[4].to(device))
                pass

            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            # (a slight speedup: https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
            #model.zero_grad()
            for weight in model.parameters():
                weight.grad = None
            outputs = model(b_input_ids, 
                attention_vals=b_att_val,
                attention_mask=b_input_mask, 
                labels=b_labels,
                device=device)

            # The call to `model` always returns a tuple, so we need to pull the 
            # loss value out of the tuple.
            
            loss = outputs[0]
           
            if(params['logging']=='neptune'):
            	neptune.log_metric('batch_loss',loss.item())
            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value 
            # from the tensor.
            total_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()
            # Update the learning rate.
            if(params['bert_tokens']):
                scheduler.step()
        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)
        if(params['logging']=='neptune'):
            neptune.log_metric('avg_train_loss',avg_train_loss)
        else:
            print('avg_train_loss',avg_train_loss)

        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)
        if epoch_i > 2:
            if(params['bert_tokens']):
                print('Loading BERT tokenizer...')
                tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=False)
                save_bert_model(model,tokenizer,params)
            else:
                print("Saving model")
                save_normal_model(model,params)
            #train_fscore,train_accuracy,train_precision,train_recall,train_roc_auc,_=Eval_phase(params,'train',model,train_dataloader,device)
            val_fscore,val_accuracy,val_precision,val_recall,val_roc_auc,_=Eval_phase(params,'val',model,validation_dataloader,device)
            test_fscore,test_accuracy,test_precision,test_recall,test_roc_auc,logits_all_final=Eval_phase(params,'test',model,test_dataloader,device)

            #Report the final accuracy for this validation run.
            if(params['logging']=='neptune'):
                neptune.log_metric('test_fscore',test_fscore)
                neptune.log_metric('test_accuracy',test_accuracy)
                neptune.log_metric('test_precision',test_precision)
                neptune.log_metric('test_recall',test_recall)
                neptune.log_metric('test_rocauc',test_roc_auc)

                neptune.log_metric('val_fscore',val_fscore)
                neptune.log_metric('val_accuracy',val_accuracy)
                neptune.log_metric('val_precision',val_precision)
                neptune.log_metric('val_recall',val_recall)
                neptune.log_metric('val_rocauc',val_roc_auc)

                # neptune.log_metric('train_fscore',train_fscore)
                # neptune.log_metric('train_accuracy',train_accuracy)
                # neptune.log_metric('train_precision',train_precision)
                # neptune.log_metric('train_recall',train_recall)
                # neptune.log_metric('train_rocauc',train_roc_auc)




            if(val_fscore > best_val_fscore):
                print(val_fscore,best_val_fscore)
                best_val_fscore=val_fscore
                best_test_fscore=test_fscore
                best_val_roc_auc = val_roc_auc
                best_test_roc_auc = test_roc_auc


                best_val_precision = val_precision
                best_test_precision = test_precision
                best_val_recall = val_recall
                best_test_recall = test_recall

                # if(params['bert_tokens']):
                #     print('Loading BERT tokenizer...')
                #     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=False)
                #     save_bert_model(model,tokenizer,params)
                # else:
                #     print("Saving model")
                #     save_normal_model(model,params)

    if(params['logging']=='neptune'):
        neptune.log_metric('best_val_fscore',best_val_fscore)
        neptune.log_metric('best_test_fscore',best_test_fscore)
        neptune.log_metric('best_val_rocauc',best_val_roc_auc)
        neptune.log_metric('best_test_rocauc',best_test_roc_auc)
        neptune.log_metric('best_val_precision',best_val_precision)
        neptune.log_metric('best_test_precision',best_test_precision)
        neptune.log_metric('best_val_recall',best_val_recall)
        neptune.log_metric('best_test_recall',best_test_recall)
        
        neptune.stop()
    else:
        print('best_val_fscore',best_val_fscore)
        print('best_test_fscore',best_test_fscore)
        print('best_val_rocauc',best_val_roc_auc)
        print('best_test_rocauc',best_test_roc_auc)
        print('best_val_precision',best_val_precision)
        print('best_test_precision',best_test_precision)
        print('best_val_recall',best_val_recall)
        print('best_test_recall',best_test_recall)
        
    del model
    torch.cuda.empty_cache()
    return 1









params_data={
    'include_special':False, 
    'bert_tokens':False,
    'type_attention':'softmax',
    'set_decay':0.1,
    'majority':2,
    'max_length':128,
    'variance':10,
    'window':4,
    'alpha':0.5,
    'p_value':0.8,
    'method':'additive',
    'decay':False,
    'normalized':False,
    'not_recollect':True,
}

#"birnn","birnnatt","birnnscrat","cnn_gru"


common_hp={
    'is_model':True,
    'logging':'local',  ###neptune /local
    'learning_rate':0.1,  ### learning rate 2e-5 for bert 0.001 for gru
    'epsilon':1e-8,
    'batch_size':16,
    'to_save':True,
    'epochs':10,
    'auto_weights':True,
    'weights':[1.0,1.0,1.0],
    'model_name':'birnnscrat',
    'random_seed':42,
    'num_classes':3,
    'att_lambda':100,
    'device':'cuda',
    'train_att':True

}
    
    
params_bert={
    'path_files':'bert-base-uncased',
    'what_bert':'weighted',
    'save_only_bert':False,
    'supervised_layer_pos':11,
    'num_supervised_heads':1,
    'dropout_bert':0.1
 }


params_other = {
        "vocab_size": 0,
        "padding_idx": 0,
        "hidden_size":64,
        "embed_size":0,
        "embeddings":None,
        "drop_fc":0.2,
        "drop_embed":0.2,
        "drop_hidden":0.1,
        "train_embed":False,
        "seq_model":"gru",
        "attention":"softmax"
}


if(params_data['bert_tokens']):
    for key in params_other:
        params_other[key]='N/A'
else:
    for key in params_bert:
        params_bert[key]='N/A'


def Merge(dict1, dict2,dict3, dict4): 
    res = {**dict1, **dict2,**dict3, **dict4} 
    return res 

params = Merge(params_data,common_hp,params_bert,params_other)


dict_data_folder={
      '2':{'data_file':'Data/dataset.json','class_label':'Data/classes_two.npy'},
      '3':{'data_file':'Data/dataset.json','class_label':'Data/classes.npy'}
}

if __name__=='__main__': 
    my_parser = argparse.ArgumentParser(description='Train a deep-learning model with the given data')

    # Add the arguments
    my_parser.add_argument('path',
                           metavar='--path_to_json',
                           type=str,
                           help='The path to json containing the parameters')
    
    my_parser.add_argument('use_from_file',
                           metavar='--use_from_file',
                           type=str,
                           help='whether use the parameters present here or directly use from file')
    
    my_parser.add_argument('attention_lambda',
                           metavar='--attention_lambda',
                           type=str,
                           help='required to assign the contribution of the attention loss')
    
    
    
    args = my_parser.parse_args()
    params['best_params']=False
    if(args.use_from_file == 'True'):
        with open(args.path,mode='r') as f:
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
        params['best_params']=True 
    ##### change in logging to output the results to neptune
    params['logging']='local'
    if(params['logging']=='neptune'):
        from api_config import project_name,api_token
        neptune.init(project_name,api_token=api_token)
        neptune.set_project(project_name)
    torch.autograd.set_detect_anomaly(True)
    if torch.cuda.is_available() and params['device']=='cuda':    
        # Tell PyTorch to use the GPU.    
        device = torch.device("cuda")
        ##### You can set the device manually if you have only one gpu
        ##### comment this line if you don't want to manually set the gpu
        deviceID = get_gpu()
        torch.cuda.set_device(deviceID[0])
        ##### comment this line if you don't want to manually set the gpu
        #### parameter required is the gpu id
        #torch.cuda.set_device(0)
        print("Using %s" % deviceID[0])
        
    else:
        print('Since you dont want to use GPU, using the CPU instead.')
        device = torch.device("cpu")
        
        
    #### Few handy keys that you can directly change.
    params['variance']=1
    params['epochs']=5
    params['to_save']=True
    params['num_classes']=3
    params['data_file']=dict_data_folder[str(params['num_classes'])]['data_file']
    params['class_names']=dict_data_folder[str(params['num_classes'])]['class_label']
    if(params['num_classes']==2 and (params['auto_weights']==False)):
          params['weights']=[1.0,1.0]
            
    #for att_lambda in [0.001,0.01,0.1,1,10,100]
    params['att_lambda']=float(args.attention_lambda)
    train_model(params,device)

















