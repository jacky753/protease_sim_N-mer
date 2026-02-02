#protein-bert-seq_RF-regressor_ALL-Human-proteases_into_4-proteases_for_mutations_16-mer
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import csv
from pandas import plotting 
import urllib.request 
import matplotlib.ticker as ticker
import sklearn #機械学習のライブラリ
from sklearn.decomposition import PCA #主成分分析器
from proteinbert import load_pretrained_model
from proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs
from msilib import Feature
from tkinter import XView
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import urllib.request
import sys
import os, glob
from sklearn.model_selection import KFold
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
#from pytorchtools import EarlyStopping
#early_stopping = EarlyStopping(patience=3, verbose=True)
import torch.optim as optim
#import optuna
import json
import statistics
import math
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from torch.nn import LayerNorm
from torch.utils.data import Dataset, DataLoader
from torch.nn import TransformerEncoder, TransformerDecoder, TransformerEncoderLayer, TransformerDecoderLayer
import datetime
import scipy
from scipy import signal
from urllib.request import urlopen
from lxml import etree
import itertools
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from matplotlib.colors import ListedColormap
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from sklearn.datasets import load_iris
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import MySQLdb as mydb
import time
import datetime

charge = {"A":0., "C":-0.0735876, "D":-0.9994991, "E":-0.9987427, "F":0., "G":0., "H":0.0593509, "I":0.,
          "K":0.9997489, "L":0., "M":0., "N":0., "P":0., "Q":0., "R":0.9999950, "S":0., "T":0., "V":0., "W":0., "Y":-0.0001995}
hydrophobicity = {"A":0.1630295, "C":-0.2554557, "D":-0.9794608, "E":-0.7458280, "F":0.9152760, "G":-0.2554557, "H":-0.7869063, "I":0.8510911, "K":-0.8510911, "L":1.,
                  "M":0.7227214, "N":-1., "P":-0.0860077, "Q":-0.8305520, "R":-0.7021823, "S":-0.4685494, "T":-0.3838254, "V":0.6816431, "W":0.8305520, "Y":0.5738126}

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
class_list = ["Negac", "Posic", "Noc", "Nonp"]
Negac_list = ["D", "E"]
Posic_list = ["R", "K", "H"]
Noc_list = ["N", "Q", "S", "T", "Y"]
Nonp_list = ["A", "G", "V", "L", "I", "P", "F", "M", "W", "C"]

def main():
    encoding_mode = 'pgbo'
    trim_num = 160
    cur = con_db()
    merops_code_mece = create_merops_code_table()
    print("merops_code_mece.")
    #print(merops_code_mece)
    print(len(merops_code_mece))
    n_mer = 160#16
    
    #for k, protease in enumerate(['S01.047', 'S01.021', 'S01.247', 'S01.034', 'S01.087']): 
    protease_num_dic = {'S01.247': 566, 'S08.071': 741}
    #for i, protease in enumerate(['S01.247']): 
    for i, protease in enumerate(['S08.071']): 
        #i = i + protease_num_dic['S01.247']
        i = i + protease_num_dic['S08.071']
        print("i: {}".format(i))
        dir_path = "./outputdir_rf_reg_priority/{}/{}_{}".format(encoding_mode, i, protease)
        # ディレクトリが存在しなければ作成する
        os.makedirs(dir_path, exist_ok=True)
        print("Directory OK")
        print("="*10+"protease: {}".format(protease)+"_protease_turn: {}".format(i)+"="*10)
        merops_code = [protease]
        cur.execute("SELECT uniprot_acc, p1 FROM cleavage where code=(%s);", merops_code)
        subs = cur.fetchall()
        print("="*10+"encoding data"+"="*10)
        pretrained_model_generator02, input_encoder02 = load_pretrained_model()
        df_positive = pd.read_csv('./proteases/cleave_pattern_one_letter_aa_{}.csv'.format(protease))

        seqs02= df_positive['cleave_pattern']

        print("seqs02.")
        print(seqs02)
        seq_len02= len(seqs02[0])+2
        print("seq_len")
        print(seq_len02)

        batch_size = 4#1
        model02 = get_model_with_hidden_layers_as_outputs(pretrained_model_generator02.create_model(seq_len02))
        print("model02.")
        print(model02)
        
        encoded_x_positive02 = input_encoder02.encode_X(seqs02, seq_len02)
        local_representations_positive, global_representations_positive = model02.predict(encoded_x_positive02, batch_size=batch_size)

        pretrained_model_generator03, input_encoder03 = load_pretrained_model()
        df_negative =  pd.read_csv('./proteases/negative_pattern_one_letter_aa_{}.csv'.format(protease))

        seqs03= df_negative['negative_pattern']
        print("seqs03 on negative data.")
        print(seqs03)
        seq_len03 = len(seqs03[0])+2
        print("seq_len03.")
        print(seq_len03)
        for j in range(len(seqs03)):
            seqs03.iloc[j] = seqs03.iloc[j].replace('-', '')
        model03 = get_model_with_hidden_layers_as_outputs(pretrained_model_generator03.create_model(seq_len03))

        encoded_x_negative03 = input_encoder03.encode_X(seqs03, seq_len03)
        
        print("seqs03 in negative.")
        print(len(seqs03))
        print("seq_len03: {}".format(seq_len03))
        print("Try model.predict() in negative data.")
        local_representations_negative, global_representations_negative = model03.predict(encoded_x_negative03, batch_size=batch_size)
        print("local_representations_negative.")
        print(local_representations_negative.shape)

        cleave_pattern01 = df_positive['cleave_pattern'][0]#df_01.iloc[1, 3]
        uniprot_id01 = df_positive['uniprot_id'][0]
        cleave_pattern01_pb = df_positive['cleave_pattern'][0]
        uniprot_id01_pb = df_positive['uniprot_id'][0]

        print('cleave_pattern01.')
        print(cleave_pattern01)
        df_test = char2vec(cleave_pattern01, n_mer)

        test_array = df_test.to_numpy().flatten()
        df_concat = pd.DataFrame([test_array], index=['{}'.format(uniprot_id01+'+')])
        df_concat['label'] = 1

        test_array_pb_seq = local_representations_positive[0].flatten()
        df_concat_pb_seq = pd.DataFrame([test_array_pb_seq], index=['{}'.format(uniprot_id01+'+')])
        df_concat_pb_seq['label'] = 1

        test_array_pb_go = global_representations_positive[0].flatten()
        df_concat_pb_go = pd.DataFrame([test_array_pb_go], index=['{}'.format(uniprot_id01+'+')])
        df_concat_pb_go['label'] = 1


        for j in range(len(df_positive)):
            print("="*10 + "positive encoding {}/{}".format(j+1, len(df_positive)) + "="*10)
            uniprot_id = df_positive['uniprot_id'][j]
            cleave_pattern = df_positive['cleave_pattern'][j]
            one_hot_seq = char2vec(cleave_pattern, n_mer)
            one_hot_array = one_hot_seq.to_numpy().flatten()
            pb_seq_array =  local_representations_positive[j].flatten()
            pb_go_array =  global_representations_positive[j].flatten()

            df = pd.DataFrame([one_hot_array], index=['{}'.format(uniprot_id)])
            df['label'] = 1
            df_pb_seq = pd.DataFrame([pb_seq_array], index=['{}'.format(uniprot_id)])
            df_pb_seq['label'] = 1
            df_pb_go = pd.DataFrame([pb_go_array], index=['{}'.format(uniprot_id)])
            df_pb_go['label'] = 1
            df_concat = pd.concat([df_concat, df])
            df_concat_pb_seq = pd.concat([df_concat_pb_seq, df_pb_seq])
            df_concat_pb_go = pd.concat([df_concat_pb_go, df_pb_go])

        print("df_concat.")
        df_concat = df_concat.drop(index='{}'.format(uniprot_id01+'+'))
        print("i: {}".format(i))
        df_concat_pb_seq = df_concat_pb_seq.drop(index='{}'.format(uniprot_id01+'+'))
        df_concat_pb_go = df_concat_pb_go.drop(index='{}'.format(uniprot_id01+'+'))

        # create negative data
        df_concat_negative = pd.DataFrame([test_array], index=['{}'.format(uniprot_id01+'+')])
        df_concat_negative['label'] = 0 

        test_array_pb_seq = local_representations_negative[0].flatten()
        df_concat_pb_seq_negative = pd.DataFrame([test_array_pb_seq], index=['{}'.format(uniprot_id01+'+')])
        df_concat_pb_seq_negative['label'] = 0

        test_array_pb_go = global_representations_negative[0].flatten()
        df_concat_pb_go_negative = pd.DataFrame([test_array_pb_go], index=['{}'.format(uniprot_id01+'+')])
        df_concat_pb_go_negative['label'] = 0

        for j in range(len(df_negative)):
            print("="*10 + "negative encoding {}/{}".format(j+1, len(df_negative)) + "="*10)
            uniprot_id = df_negative['uniprot_id'][j]
            
            print("uniprot_id: {}".format(uniprot_id))
            negative_pattern = df_negative['negative_pattern'][j]

            print("cleave_pattern for df_concat_negative.: {}".format(negative_pattern))
            one_hot_seq = char2vec(negative_pattern, n_mer)
            one_hot_array = one_hot_seq.to_numpy().flatten()
            pb_seq_array =  local_representations_negative[j].flatten()
            pb_go_array =  global_representations_negative[j].flatten()
            
            df = pd.DataFrame([one_hot_array], index=['{}'.format(uniprot_id)])
            df['label'] = 0   
            df_pb_seq = pd.DataFrame([pb_seq_array], index=['{}'.format(uniprot_id)])
            df_pb_seq['label'] = 0
            df_pb_go = pd.DataFrame([pb_go_array], index=['{}'.format(uniprot_id)])
            df_pb_go['label'] = 0

            df_concat_pb_seq_negative = pd.concat([df_concat_pb_seq_negative, df_pb_seq])
            df_concat_pb_go_negative = pd.concat([df_concat_pb_go_negative, df_pb_go])
            df_concat_negative = pd.concat([df_concat_negative, df])
        
        df_concat_negative = df_concat_negative.drop(index='{}'.format(uniprot_id01+'+'))

        df_concat_pb_seq_negative = df_concat_pb_seq_negative.drop(index='{}'.format(uniprot_id01+'+'))

        df_concat_pb_go_negative = df_concat_pb_go_negative.drop(index='{}'.format(uniprot_id01+'+'))

        device = set_device()
        data_columns_positive = len(df_concat_pb_go.columns)-1

        x_data_positive = df_concat_pb_go.iloc[:, 0:data_columns_positive]

        y_data_positive = df_concat_pb_go.iloc[:, data_columns_positive]

        data_columns_negative = len(df_concat_pb_go_negative.columns)-1
        x_data_negative = df_concat_pb_go_negative.iloc[:, 0:data_columns_negative]
        print("x_data_negative.")
        y_data_negative = df_concat_pb_go_negative.iloc[:, data_columns_negative]
        print("y_data_negative.")
        
        seed = 42
        rng = np.random.default_rng(seed=42)
        indices_positive = rng.permutation(len(x_data_positive))
        n_positive = len(x_data_positive) // 5
        picked_idx_posi = indices_positive[:n_positive]
        remaining_idx_posi = indices_positive[n_positive:]
        x_test_positive = x_data_positive.iloc[picked_idx_posi, :]
        y_test_positive = y_data_positive[picked_idx_posi]
        x_learn_positive = x_data_positive.iloc[remaining_idx_posi, :]
        y_learn_positive = y_data_positive[remaining_idx_posi]

        indices_negative = rng.permutation(len(x_data_negative))
        n_negative = len(x_data_negative) // 5
        picked_idx_nega = indices_negative[:n_negative]
        remaining_idx_nega = indices_negative[n_negative:]
        x_test_negative = x_data_negative.iloc[picked_idx_nega, :]
        y_test_negative = y_data_negative[picked_idx_nega]
        x_learn_negative = x_data_negative.iloc[remaining_idx_nega, :]
        y_learn_negative = y_data_negative[remaining_idx_nega]

        print("x_learn_positive.shape:", x_learn_positive.shape)
        print("x_learn_negative.shape:", x_learn_negative.shape)

        print("len of x_data_positive: {}".format(len(x_data_positive)))
        print("len of x_data_negative: {}".format(len(x_data_negative)))

        print("len of x_learn_positive: {}".format(len(x_learn_positive)))
        print("len of x_learn_negative: {}".format(len(x_learn_negative)))

        x_learn = np.concatenate([x_learn_positive, x_learn_negative], axis=0)
        x_test = np.concatenate([x_test_positive, x_test_negative], axis=0)

        y_learn = np.concatenate([y_learn_positive, y_learn_negative], axis=0)
        y_test = np.concatenate([y_test_positive, y_test_negative], axis=0)
       
        forest = RandomForestRegressor()

        x_seq_positive = df_positive['cleave_pattern'].tolist()
        x_seq_negative = df_negative['negative_pattern'].tolist()
        x_seq_test_positive = []
        x_seq_test_negative = []
        #x_seq_positive_cp = x_seq_positive
        #x_seq_negative_cp = x_seq_negative
        for j in range(len(picked_idx_posi)):
            #picked_item = x_seq_positive[picked_idx_posi[j]]
            popped_item = x_seq_positive.pop(picked_idx_posi[j]-j)
            x_seq_test_positive.append(popped_item)
        for j in range(len(picked_idx_nega)):
            #picked_item = x_seq_negative[picked_idx_nega[j]]
            popped_item = x_seq_negative.pop(picked_idx_nega[j]-j)
            x_seq_test_negative.append(popped_item)
        x_test_seq_list = x_seq_test_positive + x_seq_test_negative
        x_learn_seq_list = x_seq_positive + x_seq_negative

        df_test = pd.DataFrame({
            'X_test': x_test_seq_list, 
            'y_test': y_test
        }).to_csv("./outputdir_rf_reg_priority/{}/{}_{}/df_test_{}_seqs.csv".format(encoding_mode, i, protease, protease))
        df_learn = pd.DataFrame({
            'x_learn': x_learn_seq_list, 
            'y_learn': y_learn
        }).to_csv("./outputdir_rf_reg_priority/{}/{}_{}/learn-data_{}_seqs.csv".format(encoding_mode, i, protease, protease))
        #X_seq = pd.concat([df_positive['cleave_pattern'], df_negative['negative_pattern']], ignore_index=True) 
        #X_seq_list = X_seq.tolist()
        x_seq_list = x_seq_positive + x_seq_negative

        print("len of df_positive: {}".format(len(df_positive)))
        print("len of df_negative: {}".format(len(df_negative)))


        # kf = KFold(n_splits=5, shuffle=True, random_state=42)
        #kf = KFold(n_splits=5, shuffle=True, random_state=42)
        stratifiedKFold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        print("stratifiedKFold: ")
        print(stratifiedKFold)
        #print(len(stratifiedKFold))
        #print("stratifiedKFold.split(X, y): ")
        #print(stratifiedKFold.split(X, y))

        m = 0

        for train_index, test_index in stratifiedKFold.split(x_learn, y_learn):
            roc_cv_list = []
            x_train_list = []
            y_train_list = []
            x_val_list = []
            y_val_list = []
            x_train_seq_list = []
            x_val_seq_list = []


            x_train, x_val = x_learn[train_index], x_learn[test_index]
            x_train = x_train.tolist()
            x_val = x_val.tolist()
            print("train_index: ")
            print(train_index)
            print(len(train_index))
            print("test_index: ")
            print(test_index)
            print(len(test_index))
            #X_train_seq, X_test_seq = X_seq.iloc[train_index[0]: train_index[1], 0], X_seq[test_index[0]: test_index[1], 0]
            
            y_train, y_val = y_learn[train_index], y_learn[test_index]

            x_train_list.append(x_train)
            y_train_list.append(y_train)
            x_val_list.append(x_val)
            y_val_list.append(y_val)

            print("#"*10 + "make x_train_seq_list" + "#"*10)
            for j in range(len(train_index)):
                #print("train_index[{}]: {}".format(j, train_index[j]))
                x_train_seq_list.append(x_learn_seq_list[train_index[j]])
            print("x_train_seq_list: ")
            print(x_train_seq_list)
            print(len(x_train_seq_list))

            print("#"*10 + "make x_val_seq_list" + "#"*10)
            for j in range(len(test_index)):
                #print("test_index[{}]: {}".format(j, test_index[j]))
                x_val_seq_list.append(x_learn_seq_list[test_index[j]])
            print("x_val_seq_list: ")
            print(x_val_seq_list)
            print(len(x_val_seq_list))
            


            print("#"*10 + "m: {}".format(m) + "#"*10)
            print("x_train: ")
            print(len(x_train))
            # print(len(X_test))
            print("x_val: ")
            print(len(x_val))
            # print(len(X_test))
            print("y_train: ")
            print(y_train)
            print(len(y_train))
            print("y_val: ")
            print(y_val)
            print(len(y_val))

            df_train = pd.DataFrame({
                'x_train': x_train_seq_list, 
                'y_train': y_train
            }).to_csv("./outputdir_rf_reg_priority/{}/{}_{}/df_train{}_cv{}.csv".format(encoding_mode, i, protease, protease, m))
            df_val = pd.DataFrame({
                'x_val': x_val_seq_list,  
                'y_val': y_val
            }).to_csv("./outputdir_rf_reg_priority/{}/{}_{}/df_test{}_cv{}.csv".format(encoding_mode, i, protease, protease, m))

            # Train and evaluate your model here
            forest.fit(x_train, y_train)
            #df_feature_importances = pd.DataFrame(forest.feature_importances_, 
            #                            columns=["feature_importances"], 
            #                            index=list(range(len(forest.feature_importances_))))
            # df_feature_importances.to_csv("./outputdir_rf_reg_priority/{}_{}/df_feature_importances_{}_cv{}.csv".format(i, protease, protease, m))
            
            x_val = x_val
            y_val = y_test
            y_true = y_test
            print("y_true: ")
            print(y_true)
            print(len(y_true))
            y_score = forest.predict(x_val)
            print("y_score: ")
            print(y_score)
            print(len(y_score))
            
            fpr, tpr, thresholds = roc_curve(y_true, y_score)
            roc_auc_value = roc_auc_score(y_true, y_score)
            roc_cv_list.append(roc_auc_value)

            print("threshholds: {}".format(thresholds))
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
            df_optimal_threshold = pd.DataFrame({
                'optimal_threshold': [optimal_threshold]
            }).to_csv("./outputdir_rf_reg_priority/{}/{}_{}/df_optimal_threshold_{}_cv{}.csv".format(encoding_mode, i, protease, protease, m))
            y_pred = (y_score >= optimal_threshold).astype(int)

            f1_value = f1_score(y_true, y_pred)
            precision_value = precision_score(y_true, y_pred)
            recall_value = recall_score(y_true, y_pred)
            accuracy_value = accuracy_score(y_true, y_pred)

            df_roc_auc_f1score = pd.DataFrame({
                'roc_auc': [roc_auc_value], 
                'f1_score': [f1_value], 
                'precision_score': [precision_value], 
                'recall_score': [recall_value], 
                'accuracy_value': [accuracy_value]
            }).to_csv('./outputdir_rf_reg_priority/{}/{}_{}/roc_auc_f1score_{}_cv{}.csv'.format(encoding_mode, i, protease, protease, m))
            
            plt.plot(fpr, tpr, marker='o')
            plt.xlabel('FPR: False positive rate')
            plt.ylabel('TPR: True positive rate')
            plt.title('ROC_{}'.format(protease))
            plt.grid()
            plt.savefig('./outputdir_rf_reg_priority/{}/{}_{}/roc_curve_{}_cv{}.png'.format(encoding_mode, i, protease, protease, m))
            plt.clf()
            plt.close()
            

            #print("predicteds: ")
            #print(predicteds)
            #print(type(predicteds))
            #print(len(predicteds))
            #print(predicteds.shape)

            # y_val = y_test
            df_y_val = pd.DataFrame(y_val, columns=["y_val"], index=list(range(len(y_val))))
            df_val_predicteds = pd.DataFrame(y_score, columns=["predicteds_val"], index=list(range(len(y_score))))
            #df_val_predicteds = pd.DataFrame([y_val, predicteds], columns=["y_val", "val_predict"], index=list(range(len(predicteds))))
            df_val_predicteds = pd.concat([df_y_val, df_val_predicteds], axis=1)
            df_val_predicteds.to_csv("./outputdir_rf_reg_priority/{}/{}_{}/df_val_predicteds_{}_cv{}.csv".format(encoding_mode, i, protease, protease, m))
            
            # 精度検証
            mae = mean_absolute_error(y_true, y_score)
            print("mae: {}".format(mae))
            mse = mean_squared_error(y_true, y_score)
            print("mse: {}".format(mse))
            df_val_error = pd.DataFrame([[mae, mse]], columns=["val_mae", "val_mse"], index=["0"])
            #df_val_error = pd.DataFrame([[mae], [mse]], columns=["0"], index=["val_mae", "val_mse"])
            df_val_error.to_csv("./outputdir_rf_reg_priority/{}/{}_{}/df_val_error_{}_cv{}.csv".format(encoding_mode, i, protease, protease, m))
            m += 1 




        #forest = RandomForestClassifier(criterion='entropy',
        #                            n_estimators=10, random_state=1, n_jobs=2)

        #x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.33, random_state=42)
        # x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.33, random_state=42)
        forest.fit(x_learn, y_learn)
        y_score = forest.predict(x_test)
        y_true = y_test

        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        roc_auc_value = roc_auc_score(y_true, y_score)
        roc_cv_list.append(roc_auc_value)

        print("threshholds: {}".format(thresholds))
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        df_optimal_threshold = pd.DataFrame({
            'optimal_threshold': [optimal_threshold]
        }).to_csv("./outputdir_rf_reg_priority/{}/{}_{}/df_optimal_threshold_{}_test.csv".format(encoding_mode, i, protease, protease))
        y_pred = (y_score >= optimal_threshold).astype(int)

        f1_value = f1_score(y_true, y_pred)
        precision_value = precision_score(y_true, y_pred)
        recall_value = recall_score(y_true, y_pred)
        accuracy_value = accuracy_score(y_true, y_pred)

        df_roc_auc_f1score = pd.DataFrame({
            'roc_auc': [roc_auc_value], 
            'f1_score': [f1_value], 
            'precision_score': [precision_value], 
            'recall_score': [recall_value], 
            'accuracy_value': [accuracy_value]
        }).to_csv('./outputdir_rf_reg_priority/{}/{}_{}/roc_auc_f1score_{}_test.csv'.format(encoding_mode, i, protease, protease))
        
        plt.plot(fpr, tpr, marker='o')
        plt.xlabel('FPR: False positive rate')
        plt.ylabel('TPR: True positive rate')
        plt.title('ROC_{}'.format(protease))
        plt.grid()
        plt.savefig('./outputdir_rf_reg_priority/{}/{}_{}/roc_curve_{}_test.png'.format(encoding_mode, i, protease, protease))
        plt.clf()
        plt.close()
        

        #print("predicteds: ")
        #print(predicteds)
        #print(type(predicteds))
        #print(len(predicteds))
        #print(predicteds.shape)

        # y_val = y_test
        df_y_val = pd.DataFrame(y_val, columns=["y_val"], index=list(range(len(y_val))))
        df_val_predicteds = pd.DataFrame(y_score, columns=["predicteds_val"], index=list(range(len(y_score))))
        #df_val_predicteds = pd.DataFrame([y_val, predicteds], columns=["y_val", "val_predict"], index=list(range(len(predicteds))))
        df_val_predicteds = pd.concat([df_y_val, df_val_predicteds], axis=1)
        df_val_predicteds.to_csv("./outputdir_rf_reg_priority/{}/{}_{}/df_val_predicteds_{}_test.csv".format(encoding_mode, i, protease, protease))
        
        # 精度検証
        mae = mean_absolute_error(y_true, y_score)
        print("mae: {}".format(mae))
        mse = mean_squared_error(y_true, y_score)
        print("mse: {}".format(mse))
        df_val_error = pd.DataFrame([[mae, mse]], columns=["val_mae", "val_mse"], index=["0"])
        #df_val_error = pd.DataFrame([[mae], [mse]], columns=["0"], index=["val_mae", "val_mse"])
        df_val_error.to_csv("./outputdir_rf_reg_priority/{}/{}_{}/df_val_error_{}_test.csv".format(encoding_mode, i, protease, protease))

        """
        print("#"*20 + "mostAccuracyModel" + "#"*20)
        roc_max = -1
        roc_index = -1
        for j in range(len(roc_cv_list)):
            if roc_cv_list[j] > roc_max:
                roc_index = j
            else:
                pass
        # }).to_csv("./outputdir_rf_reg_priority/{}_{}/df_train_test_{}_cv{}.csv".format(i, protease, protease, m))
        #df_learning_data = pd.read_csv("./outputdir_rf_reg_priority/{}_{}/df_train_test_{}_cv{}.csv".format(i, protease, protease, roc_index))
        df_train = pd.read_csv("./outputdir_rf_reg_priority/{}/{}_{}/df_train{}_cv{}.csv".format(encoding_mode, i, protease, protease, roc_index))
        df_test = pd.read_csv("./outputdir_rf_reg_priority/{}/{}_{}/df_test{}_cv{}.csv".format(encoding_mode, i, protease, protease, roc_index))
        
        #x_train = df_train['X_train'].tolist()
        x_train = x_train_list[roc_index]
        print("len of x_train: {}".format(len(x_train)))
        #x_train = [float(s) for s in x_train]

        #x_val = df_test['X_test'].tolist()
        x_val = x_val_list[roc_index]
        print("len of x_val: {}".format(len(x_val)))
        #x_val = [float(s) for s in x_val]

        #y_train = df_train['y_train'].tolist
        y_train = y_train_list[roc_index]
        print("len of y_train: {}".format(len(y_train)))
        #y_train = [float(s) for s in y_train]
        
        #y_val = df_test['y_test'].tolist()
        y_val = y_val_list[roc_index]
        print("len of y_val: {}".format(len(y_val)))
        #y_val = [float(s) for s in y_val]

        #print("END before sys.exit().")
        #sys.exit()

        forest = RandomForestRegressor()
        if len(df_positive) == 1 or len(df_positive) == 2:
            forest.fit(x_data, y_data)
        else:
            forest.fit(x_train, y_train)
        #print("forest.feature_importances_: ")
        #print(forest.feature_importances_)
        df_feature_importances = pd.DataFrame(forest.feature_importances_, 
                                                columns=["feature_importances"], 
                                                index=list(range(len(forest.feature_importances_))))
        # df_feature_importances.to_csv("./outputdir_rf_reg_priority/{}_{}/df_feature_importances_{}_selectedLearningData.csv".format(i, protease, protease))
        predicteds = forest.predict(x_val)
        y_true = y_val
        y_score = predicteds
        print("y_score in selected learning data: ")
        print(y_score)
        print(len(y_score))

        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        roc_auc_value = roc_auc_score(y_true, y_score)

        #print("threshholds: {}".format(thresholds))
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        y_pred = (y_score >= optimal_threshold).astype(int)

        f1_value = f1_score(y_true, y_pred)
        precision_value = precision_score(y_true, y_pred)
        recall_value = recall_score(y_true, y_pred)
        accuracy_value = accuracy_score(y_true, y_pred)

        df_roc_auc_f1score = pd.DataFrame({
            'roc_auc': [roc_auc_value], 
            'f1_score': [f1_value], 
            'precision_score': [precision_value], 
            'recall_score': [recall_value], 
            'accuracy_value': [accuracy_value]
        }).to_csv('./outputdir_rf_reg_priority/{}_{}/roc_auc_f1score_{}_selectedLearningData_roc-index{}.csv'.format(i, protease, protease, roc_index))
        
        plt.plot(fpr, tpr, marker='o')
        plt.xlabel('FPR: False positive rate')
        plt.ylabel('TPR: True positive rate')
        plt.title('ROC_{}'.format(protease))
        plt.grid()
        plt.savefig('./outputdir_rf_reg_priority/{}_{}/roc_curve_{}_selectedLearningData_roc-index{}.png'.format(i, protease, protease, roc_index))
        

        print("predicteds: ")
        #print(predicteds)
        #print(type(predicteds))
        print(len(predicteds))
        print(predicteds.shape)

        df_y_val = pd.DataFrame(y_val, columns=["y_val"], index=list(range(len(y_val))))
        df_val_predicteds = pd.DataFrame(predicteds, columns=["predicteds_val"], index=list(range(len(predicteds))))
        #df_val_predicteds = pd.DataFrame([y_val, predicteds], columns=["y_val", "val_predict"], index=list(range(len(predicteds))))
        df_val_predicteds = pd.concat([df_y_val, df_val_predicteds], axis=1)
        df_val_predicteds.to_csv("./outputdir_rf_reg_priority/{}_{}/df_val_predicteds_{}_fixLearningData_roc-index{}.csv".format(i, protease, protease, roc_index))
        
        # 精度検証
        mae = mean_absolute_error(y_true, y_score)
        print("mae: {}".format(mae))
        mse = mean_squared_error(y_true, y_score)
        print("mse: {}".format(mse))
        df_val_error = pd.DataFrame([[mae, mse]], columns=["val_mae", "val_mse"], index=["0"])
        #df_val_error = pd.DataFrame([[mae], [mse]], columns=["0"], index=["val_mae", "val_mse"])
        df_val_error.to_csv("./outputdir_rf_reg_priority/{}_{}/df_val_error_{}_fixLearningData_roc-index{}.csv".format(i, protease, protease, roc_index))
        """


        #plot_decision_regions(x_combined, y_combined, classifier=forest)
        #plt.xlabel('petal length[cm]')
        #plt.ylabel('petal width [cm]')
        #plt.legend(loc='upper left')
        #plt.show()

        #train_dataloader, val_dataloader = data_loader(x_data_tr, y_data_tr, x_data_val, y_data_val)
        #print("len of train_dataloader: {}".format(len(train_dataloader)))
        #dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}


        pretrained_model_generator, input_encoder = load_pretrained_model()
        origin_fullseq = "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQDVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEHVNNSYECDIPIGAGICASYQTQTNSPRRARSVASQSIIAYTMSLGAENSVAYSNNSIAIPTNFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQEVFAQVKQIYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDCLGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNQNAQALNTLVKQLSSNFGAISSVLNDILSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPAICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVSGNCDVVIGIVNNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDDSEPVLKGVKLHYT"
        for j in range(8):
            if j == 0:
                fullseq = origin_fullseq
                mutation_name = "origin"
            elif j == 1:
                fullseq = test_data_gen_alpha()
                mutation_name = "alpha"
            elif j == 2:
                fullseq = test_data_gen_beta()
                mutation_name = "beta"
            elif j == 3:
                fullseq = test_data_gen_gamma()
                mutation_name = "gamma"
            elif j == 4:
                fullseq = test_data_gen_delta()
                mutation_name = "delta"
            elif j == 5:
                fullseq = test_data_gen_omicron()
                mutation_name = "omicron"
            elif j == 6:
                fullseq = test_data_gen_omicron_ba1()
                mutation_name = "omicron_ba1"
            elif j == 7:
                fullseq = test_data_gen_omicron_ba2()
                mutation_name = "omicron_ba2"
            """
            elif j == 7:
                fullseq = test_data_gen_omicron_ba3()
                mutation_name = "omicron_ba3"
            elif j == 8:
                fullseq = test_data_gen_omicron_jn1()
                mutation_name = "omicron_jn1"
            elif j == 9:
                fullseq = test_data_gen_omicron_eg5()
                mutation_name = "omicron_eg5"
            elif j == 10:
                fullseq = test_data_gen_omicron_kp3()
                mutation_name = "omicron_kp3"
            elif j == 11:
                fullseq = test_data_gen_omicron_xec()
                mutation_name = "omicron_xec"
            else:
                pass
            """

            spike_protein_eight_aa_list = test_data_gen(fullseq, trim_num)
            df_test_pattern = pd.DataFrame(spike_protein_eight_aa_list, columns=["test_pattern"], index=list(range(len(spike_protein_eight_aa_list))))
            seqs = df_test_pattern["test_pattern"]
            seq_len= len(seqs[0])+2

            model = get_model_with_hidden_layers_as_outputs(pretrained_model_generator.create_model(seq_len))
            encoded_x = input_encoder.encode_X(seqs, seq_len)
            local_representations, global_representations = model.predict(encoded_x, batch_size=batch_size)
            x_test = global_representations.reshape(int(global_representations.shape[0]), -1)
            
            #y_test = np.array([1.0, 0.0]).reshape(-1, 1)
            y_test = np.zeros((len(x_test), 1))
            #print(y_test)
            #print(y_test.shape)

            #test_dataloader = test_data_loader(x_test, y_test)

            #d_input = data_columns_positive
            #_output = 1 #2. this parameter is used in Decoder.
            #d_model = 64 #64.
            #nhead = 8 #2. 4. 
            #dim_feedforward = 1024   #2024. 16. 88.
            #num_encoder_layers = 1 
            #num_decoder_layers = 0
            #dropout = 0.01  #0.02. 0.1, 
            #src_len = 18 # it was 18.
            #tgt_len = 2 # it was 2. 
            #batch_size = 8
            #epochs = 5 #30, 200
            
            #best_loss = float('Inf')
            #best_model = None
            #model = Transformer(num_encoder_layers=num_encoder_layers,
            #                    num_decoder_layers=num_decoder_layers,
            #                    d_model=d_model,
            #                   d_input=d_input, 
            #                   d_output=d_output,
            #                   dim_feedforward=dim_feedforward,
            #                    dropout=dropout, nhead=nhead
            #                ).to(device)
            #model = NN(d_model, d_input, d_output) 
            #predictdata = predict(test_dataloader, model, device, model_path)
            predictdata = forest.predict(x_test)

            print("predictdata = output_all in main function.")
            #print(predictdata)
            print(predictdata.shape)
            df_predict = pd.DataFrame(predictdata, columns=["predict"], index=list(range(len(predictdata))))
            df_predict.to_csv("./outputdir_rf_reg_priority/{}_{}/predict_sprotein_{}.csv".format(i, protease, protease))
            df_concat_predict_seqs = pd.concat([df_predict, seqs], axis=1)
            df_concat_predict_seqs.to_csv("./outputdir_rf_reg_priority/{}_{}/predict_sprotein_seqs_{}_{}.csv".format(i, protease, protease, mutation_name))

"""
def con_db():
    # コネクションの作成
    conn = mydb.connect(
        host='localhost',
        port=3306,
        user='root',
        password='miyazakilab',
        #password='KtHk23#8', 
        #database='meropsrefs01'
        database='meropsweb12_1'
        #database='meropsweb121'
    )

    # DB操作用にカーソルを作成
    cur = conn.cursor()
    return cur
"""
def con_db():
    # コネクションの作成
    conn = mydb.connect(
        host='localhost',
        port=3306,
        user='root',
        #password='miyazakilab',
        password='KtHk23#8', 
        #password='QuNr#he',
        #database='meropsrefs01'
        #database='meropsweb12_1'
        database='meropsweb121'
    )

    # DB操作用にカーソルを作成
    cur = conn.cursor()
    return cur

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    #setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    # 最小値, 最大値からエリアの領域を割り出す
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # resolutionの間隔で区切った領域を定義
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                            np.arange(x2_min, x2_max, resolution))
    # print(xx1.shape)

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

   # plot all samples
    X_test, y_test = X[test_idx, :], y[test_idx]
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)

    # highlight test samples
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='',
            alpha=1.0, linewidth=1, marker='o',
            s=55, label='test set')

def tensor_to_list(tensor_list):
    concat_list = []
    for i in range(len(tensor_list)):
        temp = tensor_list[i].tolist()
        concat_list.append(temp)
    count = 0
    for item in concat_list:
        count += len(item)
    return list(itertools.chain.from_iterable(concat_list))

def create_merops_code_table():
    csv_file = open("./testdata/merops_code_mece.csv", "r", encoding="ms932", errors="", newline="" )
    #リスト形式
    f = csv.reader(csv_file, delimiter=",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)

    #print(f)
    #print(len(f))

    i = 0
    for row in f:
        print(row)
        merops_code_mece = row
        i = i + 1
        if i == 1:
            break

    #print("merops_code_mece")
    #print(merops_code_mece)
    #len(merops_code_mece)
    return merops_code_mece






arr = [[0,0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0,0]
       ]
df = pd.DataFrame(arr,
                  columns = ["P4","P3","P2", "P1", "P1dash", "P2dash", "P3dash", "P4dash"],
                  index = ["A","C","D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"])
#print("df.")
#print(df)
#print(df.shape)

#dict1 = {"P4":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#         "P3":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#         "P2":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#         "P3":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 
#         "P1":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#         "P1dash":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#         "P2dash":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#         }

#x_train = np.zeros([len(df_01), 20, len(cleave_pattern01), 1])
def char2vec(cleave_pattern, n_mer):
    #arr = [[0,0,0,0,0,0,0,0],
    #    [0,0,0,0,0,0,0,0],
    #    [0,0,0,0,0,0,0,0],
    #    [0,0,0,0,0,0,0,0],
    #    [0,0,0,0,0,0,0,0],
    #    [0,0,0,0,0,0,0,0],
    #    [0,0,0,0,0,0,0,0],
    #    [0,0,0,0,0,0,0,0],
    #    [0,0,0,0,0,0,0,0],
    #    [0,0,0,0,0,0,0,0],
    #    [0,0,0,0,0,0,0,0],
    #    [0,0,0,0,0,0,0,0],
    #    [0,0,0,0,0,0,0,0],
    #    [0,0,0,0,0,0,0,0],
    #    [0,0,0,0,0,0,0,0],
    #    [0,0,0,0,0,0,0,0],
    #    [0,0,0,0,0,0,0,0],
    #    [0,0,0,0,0,0,0,0],
    #    [0,0,0,0,0,0,0,0],
    #    [0,0,0,0,0,0,0,0]
    #    ]
    arr = [[0]*n_mer]*20
    col_num = int(n_mer//2)
    columns_list = []
    columns_p = []
    columns_p_dash = []
    for i in range(col_num):
        columns_p.append("P{}".format(col_num - i))
        columns_p_dash.append("P{}dash".format(i + 1))
    #columns_list.append(columns_p)
    #columns_list.append(columns_p_dash)
    columns_list = columns_p + columns_p_dash

    #print("columns_list.")
    #print(columns_list)
    df = pd.DataFrame(arr,
                    #columns = ["P4","P3","P2", "P1", "P1dash", "P2dash", "P3dash", "P4dash"],
                    columns = columns_list,
                    index = ["A","C","D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"])
    print("cleave_pattern: ")
    print(cleave_pattern)

    for i in range(len(cleave_pattern)): 
        #print("j in char2vec func.: {}".format(j))
        aa = cleave_pattern[i]  
        if aa == "A":
            #x_train[num][0][j] += 1.0
            df.iloc[0, i] +=1.0
        elif aa == "C":
            #x_train[num][1][j] += 1.0 
            df.iloc[1, i] +=1.0
        elif aa == "D":
            #x_train[num][2][j] += 1.0 
            df.iloc[2, i] +=1.0
        elif aa == "E":
            #x_train[num][3][j] += 1.0 
            df.iloc[3, i] +=1.0
        elif aa == "F":
            #x_train[num][4][j] += 1.0
            df.iloc[4, i] +=1.0
        elif aa == "G":
            #x_train[num][5][j] += 1.0
            df.iloc[5, i] +=1.0
        elif aa == "H":
            #x_train[num][6][j] += 1.0
            df.iloc[6, i] +=1.0
        elif aa == "I":
            #x_train[num][7][j] += 1.0
            df.iloc[7, i] +=1.0
        elif aa == "K":
            #x_train[num][8][j] += 1.0
            df.iloc[8, i] +=1.0
        elif aa == "L":
            #x_train[num][9][j] += 1.0
            df.iloc[9, i] +=1.0
        elif aa == "M":
            #x_train[num][10][j] += 1.0
            df.iloc[10, i] +=1.0
        elif aa == "N":
            #x_train[num][11][j] += 1.0
            df.iloc[11, i] +=1.0
        elif aa == "P":
            #x_train[num][12][j] += 1.0
            df.iloc[12, i] +=1.0
        elif aa == "Q":
            #x_train[num][13][j] += 1.0
            df.iloc[13, i] +=1.0
        elif aa == "R":
            #x_train[num][14][j] += 1.0
            df.iloc[14, i] +=1.0
        elif aa == "S":
            #x_train[num][15][j] += 1.0
            df.iloc[15, i] +=1.0
        elif aa == "T":
            #x_train[num][16][j] += 1.0
            df.iloc[16, i] +=1.0
        elif aa == "V":
            #x_train[num][17][j] += 1.0
            df.iloc[17, i] +=1.0
        elif aa == "W":
            #x_train[num][18][j] += 1.0
            df.iloc[18, i] +=1.0
        elif aa == "Y":
            #x_train[num][19][j] += 1.0
            df.iloc[19, i] +=1.0
        else:
            pass
    return df





def set_device():
    device_var = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("使用デバイス:", device_var)
    device = torch.device(device_var)
    return device

#def set_data_param():
#    #データ前処理 transform を設定
#    transform = transforms.Compose(
#    [transforms.ToTensor(),                      # Tensor変換とshape変換 [H, W, C] -> [C, H, W]
#        transforms.Normalize((0.5, ), (0.5, ))])    # 標準化 平均:0.5  標準偏差:0.5
#   return 1
def aaseq_from_uid(uid, protease_turn, substrate_turn):
    df = pd.DataFrame(np.arange(3).reshape(1, 3), columns=['uniprotKB_accession', 'function', 'sequence'], index=['protease'+str(protease_turn)+'_substrate'+str(substrate_turn)])

    for column_name in df:
        df[column_name] = df[column_name].astype(str)

    df['uniprotKB_accession'][0] = uid

    #display(df)    

    url = "https://www.uniprot.org/uniprot/" + uid + ".xml"
    f = urlopen(url)
    xml = f.read()
    root = etree.fromstring(xml)
    
    #以下のコードは下の説明を参照
    function = root.find('./entry/comment[@type="function"]', root.nsmap)
    if function==None:
        print("function was not detected.")
        pass
    else:
        df["function"][0] = function[0].text
        #print(function[0].text+"¥n")

    sequence = root.find('./entry/sequence', root.nsmap) 
    if sequence==None: 
        print("sequence was not detected.")
        pass 
    else: 
        df["sequence"][0] = sequence.text 
        #print(sequence.text+"¥n")

    #display(df) 
    #df0=pd.concat([df0, df], axis=0)
    #display(df0)
    print(df)
    print(df["sequence"][0]) 
    return df["sequence"][0]

def test_data_gen(fullseq, trim_num):
    #uniprot_id = "P0DTC2"
    #fullseq = aaseq_from_uid(uniprot_id, 0, 0)
    # 1273 aa
    # https://www.ncbi.nlm.nih.gov/nuccore/NC_045512.2
    #fullseq = "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQDVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEHVNNSYECDIPIGAGICASYQTQTNSPRRARSVASQSIIAYTMSLGAENSVAYSNNSIAIPTNFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQEVFAQVKQIYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDCLGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNQNAQALNTLVKQLSSNFGAISSVLNDILSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPAICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVSGNCDVVIGIVNNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDDSEPVLKGVKLHYT"
    #fullseq_list = list(fullseq)


    spike_protein_aa_list = []

    for loc_ in range(len(fullseq)-trim_num):
        eight_seq = fullseq[loc_: loc_+trim_num]
        spike_protein_aa_list.append(eight_seq)

    return spike_protein_aa_list

def test_data_gen_alpha():
    fullseq_mutation = 'MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAISGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLGVYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTYGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIDDTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQGVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEHVNNSYECDIPIGAGICASYQTQTNSHRRARSVASQSIIAYTMSLGAENSVAYSNNSIAIPINFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQEVFAQVKQIYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDCLGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNQNAQALNTLVKQLSSNFGAISSVLNDILARLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPAICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTHNTFVSGNCDVVIGIVNNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDDSEPVLKGVKLHYT'
    return fullseq_mutation

def test_data_gen_beta():
    fullseq_mutation = 'MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFANPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRGLPQGFSALEPLVDLPIGINITRFQTLHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGNIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVKGFNCYFPLQSYGFQPTYGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQGVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGVEHVNNSYECDIPIGAGICASYQTQTNSPRRARSVASQSIIAYTMSLGVENSVAYSNNSIAIPTNFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQEVFAQVKQIYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDCLGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNQNAQALNTLVKQLSSNFGAISSVLNDILSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPAICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVSGNCDVVIGIVNNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDDSEPVLKGVKLHYT'
    return fullseq_mutation

def test_data_gen_gamma():
    fullseq_mutation = 'MFVFLVLLPLVSSQCVNFTNRTQLPSAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNYPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGTIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTYGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQGVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEYVNNSYECDIPIGAGICASYQTQTNSPRRARSVASQSIIAYTMSLGAENSVAYSNNSIAIPTNFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQEVFAQVKQIYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDCLGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNQNAQALNTLVKQLSSNFGAISSVLNDILSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAAIKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPAICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVSGNCDVVIGIVNNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASFVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDDSEPVLKGVKLHYT'
    return fullseq_mutation

def test_data_gen_delta():
    fullseq_mutation = 'MFVFLVLLPLVSSQCVNLRTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASIEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLDVYYHKNNKSWMESGVYSSANNCTFEYVSQPFLMDLEVKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYRYRLFRKSNLKPFERDISTEIYQAGSKPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQGVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEHVNNSYECDIPIGAGICASYQTQTNSRRRARSVASQSIIAYTMSLGAENSVAYSNNSIAIPTNFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQEVFAQVKQIYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDCLGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQNVVNQNAQALNTLVKQLSSNFGAISSVLNDILSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPAICHDGKAHFPREGVFVSNGTHWFLTQRNFYEPQTITTDNTFVSGNCDVVIGIVNNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDDSEPVLKGVKLHYT'
    return fullseq_mutation

def test_data_gen_omicron():
    fullseq_mutation = "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFDEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVFRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGVNCYFPLQSYGFQPTYGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQGVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEYVNNSYECDIPIGAGICASYQTQTKSHRRARSVASQSIIAYTMSLGAENSVAYSNNSIAIPTNFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQEVFAQVKQIYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDCLGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNHNAQALNTLVKQLSSKFGAISSVLNDILSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPAICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVSGNCDVVIGIVNNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDDSEPVLKGVKLHYT"
    return fullseq_mutation

def test_data_gen_omicron_ba1():
    fullseq_mutation = 'MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHVISGTNGTKRFDNPVLPFNDGVYFASIEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLDHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPIIVREPEDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFDEVFNATRFASVYAWNRKRISNCVADYSVLYNFAPFFAFKCYGVSPTKLNDLCFTNVYADSFVIRGNEVSQIAPGQTGNIADYNYKLPDDFTGCVIAWNSNKLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGNKPCNGVAGFNCYFPLRSYGFRPTYGVGHQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLKGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQGVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEYVNNSYECDIPIGAGICASYQTQTKSHRRARSVASQSIIAYTMSLGAENSVAYSNNSIAIPTNFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLKRALTGIAVEQDKNTQEVFAQVKQIYKTPPIKYFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDCLGDIAARDLICAQKFKGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNHNAQALNTLVKQLSSKFGAISSVLNDIFSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPAICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVSGNCDVVIGIVNNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDDSEPVLKGVKLHYT'
    return fullseq_mutation

def test_data_gen_omicron_ba2():
    fullseq_mutation = 'MFVFLVLLPLVSSQCVNLITRTQSYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLDVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLGRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFDEVFNATRFASVYAWNRKRISNCVADYSVLYNXAPFFAFKCYGVSPTKLNDLCFTNVYADSFVIRGNEVSQIAPGQTGNIADYNYKLPDDFTGCVIAWNSNKLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGNKPCNGVAGFNCYFPLRSYGFRPTYGVGHQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQGVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEYVNNSYECDIPIGAGICASYQTQTKSHRRARSVASQSIIAYTMSLGAENSVAYSNNSIAIPTNFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLKRALTGIAVEQDKNTQEVFAQVKQIYKTPPIKYFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDCLGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNHNAQALNTLVKQLSSKFGAISSVLNDILSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPAICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVSGNCDVVIGIVNNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDDSEPVLKGVKLHYT'
    return fullseq_mutation

def test_data_gen_omicron_ba3():
    # S	changes	A67V, T95I, Y145D, L212I, G339D, S371F, S373P, S375F, D405N, K417N, N440K, G446S, S477N, T478K, E484A, Q493R, Q498R, N501Y, Y505H, D614G, H655Y, N679K, P681H, N764K, D796Y, Q954H, N969K
    # S	gaps	H69-, V70-, G142-, V143-, Y144-, N211-
    fullseq = "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQDVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEHVNNSYECDIPIGAGICASYQTQTNSPRRARSVASQSIIAYTMSLGAENSVAYSNNSIAIPTNFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQEVFAQVKQIYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDCLGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNQNAQALNTLVKQLSSNFGAISSVLNDILSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPAICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVSGNCDVVIGIVNNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDDSEPVLKGVKLHYT"
    fullseq_list = list(fullseq)
    df = pd.read_csv('./mutations/omicron_ba3.csv', header=None)
    #print(df)
    #content = r'hellow python, 123,end' 
    content_list = df.iloc[0, 1:].tolist()
    #print(content_list)
    content_list02 = df.iloc[1, 1:7].tolist()
    #print(content_list02)
    for content in content_list:
        #print("="*10)
        #print("content: {}".format(content))
        pattern = r'^\s*([A-Z])(\d+)([A-Z])$'
        result = re.match(pattern, content)
        if result: #none以外の場合
            mutation_position = int(result.group(2))
            #print("metation_position: {}".format(mutation_position))
            new_aa = result.group(3)
            #print("new_aa: {}".format(new_aa))
            #print("fullseq_list[mutation_position-1] : {}".format(fullseq_list[mutation_position-1] ))
            fullseq_list[mutation_position-1] = new_aa
    for content in content_list02:
        #print("="*10)
        #print("content: {}".format(content))
        pattern = r'^\s*([A-Z])(\d+)(.+)$'
        result = re.match(pattern, content) 
        gap_count = 0
        if result: #none以外の場合
            gap_position = int(result.group(2))
            #print("gap_position: {}".format(gap_position))
            del fullseq_list[gap_position-gap_count-1]
            gap_count += 1
    fullseq_mutation = "".join(fullseq_list)
    return fullseq_mutation

def test_data_gen_omicron_jn1():
    # S	changes	T19I, R21T, A27S, S50L, V127F, G142D, F157S, R158G, L212I, V213G, L216F, H245N, A264D, I332V, G339H, K356T, S371F, S373P, S375F, T376A, R403K, D405N, R408S, K417N, N440K, V445H, G446S, N450D, L452W, L455S, N460K, S477N, T478K, N481K, E484K, F486P, Q498R, N501Y, Y505H, E554K, A570V, D614G, P621S, H655Y, N679K, P681R, N764K, D796Y, S939F, Q954H, N969K, P1143L
    # S	reversionsToRoot	A67A, T95T, V143V, Y145Y, Q493Q
    # S	gaps	L24-, P25-, P26-, H69-, V70-, Y144-, N211-, V483-
    fullseq = "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQDVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEHVNNSYECDIPIGAGICASYQTQTNSPRRARSVASQSIIAYTMSLGAENSVAYSNNSIAIPTNFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQEVFAQVKQIYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDCLGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNQNAQALNTLVKQLSSNFGAISSVLNDILSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPAICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVSGNCDVVIGIVNNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDDSEPVLKGVKLHYT"
    fullseq_list = list(fullseq)
    df = pd.read_csv('./mutations/omicron_jn1.csv', header=None)
    #print(df)
    #content = r'hellow python, 123,end' 
    content_list = df.iloc[0, 1:].tolist()
    #print(content_list)
    content_list02 = df.iloc[2, 1:9].tolist()
    #print(content_list02)
    for content in content_list:
        #print("="*10)
        #print("content: {}".format(content))
        pattern = r'^\s*([A-Z])(\d+)([A-Z])$'
        result = re.match(pattern, content)
        if result: #none以外の場合
            mutation_position = int(result.group(2))
            #print("metation_position: {}".format(mutation_position))
            new_aa = result.group(3)
            #print("new_aa: {}".format(new_aa))
            #print("fullseq_list[mutation_position-1] : {}".format(fullseq_list[mutation_position-1] ))
            fullseq_list[mutation_position-1] = new_aa
    for content in content_list02:
        #print("="*10)
        #print("content: {}".format(content))
        pattern = r'^\s*([A-Z])(\d+)(.+)$'
        result = re.match(pattern, content) 
        gap_count = 0
        if result: #none以外の場合
            gap_position = int(result.group(2))
            #print("gap_position: {}".format(gap_position))
            del fullseq_list[gap_position-gap_count-1]
            gap_count += 1
    fullseq_mutation = "".join(fullseq_list)
    return fullseq_mutation

def test_data_gen_omicron_eg5():
    # S	changes	T19I, A27S, V83A, G142D, H146Q, Q183E, V213E, G252V, G339H, R346T, L368I, S371F, S373P, S375F, T376A, D405N, R408S, K417N, N440K, V445P, G446S, F456L, N460K, S477N, T478K, E484A, F486P, F490S, Q498R, N501Y, Y505H, D614G, S640F, H655Y, N679K, P681H, N764K, D796Y, Q954H, N969K
    # S	gaps	L24-, P25-, P26-, Y144-
    fullseq = "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQDVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEHVNNSYECDIPIGAGICASYQTQTNSPRRARSVASQSIIAYTMSLGAENSVAYSNNSIAIPTNFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQEVFAQVKQIYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDCLGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNQNAQALNTLVKQLSSNFGAISSVLNDILSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPAICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVSGNCDVVIGIVNNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDDSEPVLKGVKLHYT"
    fullseq_list = list(fullseq)
    df = pd.read_csv('./mutations/omicron_eg5.csv', header=None)
    #print(df)
    #content = r'hellow python, 123,end' 
    content_list = df.iloc[0, 1:].tolist()
    #print(content_list)
    content_list02 = df.iloc[1, 1:5].tolist()
    #print(content_list02)
    for content in content_list:
        #print("="*10)
        #print("content: {}".format(content))
        pattern = r'^\s*([A-Z])(\d+)([A-Z])$'
        result = re.match(pattern, content)
        if result: #none以外の場合
            mutation_position = int(result.group(2))
            #print("metation_position: {}".format(mutation_position))
            new_aa = result.group(3)
            #print("new_aa: {}".format(new_aa))
            #print("fullseq_list[mutation_position-1] : {}".format(fullseq_list[mutation_position-1] ))
            fullseq_list[mutation_position-1] = new_aa
    for content in content_list02:
        #print("="*10)
        #print("content: {}".format(content))
        pattern = r'^\s*([A-Z])(\d+)(.+)$'
        result = re.match(pattern, content) 
        gap_count = 0
        if result: #none以外の場合
            gap_position = int(result.group(2))
            #print("gap_position: {}".format(gap_position))
            del fullseq_list[gap_position-gap_count-1]
            gap_count += 1
    fullseq_mutation = "".join(fullseq_list)
    return fullseq_mutation

def test_data_gen_omicron_kp3():
    # S	changes	T19I, R21T, A27S, S50L, V127F, G142D, F157S, R158G, L212I, V213G, L216F, H245N, A264D, I332V, G339H, K356T, S371F, S373P, S375F, T376A, R403K, D405N, R408S, K417N, N440K, V445H, G446S, N450D, L452W, L455S, F456L, N460K, S477N, T478K, N481K, E484K, F486P, Q493E, Q498R, N501Y, Y505H, E554K, A570V, D614G, P621S, H655Y, N679K, P681R, N764K, D796Y, S939F, Q954H, N969K, V1104L, P1143L
    # S	reversionsToRoot	A67A, T95T, V143V, Y145Y
    # S	gaps	L24-, P25-, P26-, H69-, V70-, Y144-, N211-, V483-
    fullseq = "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQDVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEHVNNSYECDIPIGAGICASYQTQTNSPRRARSVASQSIIAYTMSLGAENSVAYSNNSIAIPTNFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQEVFAQVKQIYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDCLGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNQNAQALNTLVKQLSSNFGAISSVLNDILSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPAICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVSGNCDVVIGIVNNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDDSEPVLKGVKLHYT"
    fullseq_list = list(fullseq)
    df = pd.read_csv('./mutations/omicron_kp3.csv', header=None)
    #print(df)
    #content = r'hellow python, 123,end' 
    content_list = df.iloc[0, 1:].tolist()
    #print(content_list)
    content_list02 = df.iloc[2, 1:9].tolist()
    #print(content_list02)
    for content in content_list:
        #print("="*10)
        #print("content: {}".format(content))
        pattern = r'^\s*([A-Z])(\d+)([A-Z])$'
        result = re.match(pattern, content)
        if result: #none以外の場合
            mutation_position = int(result.group(2))
            #print("metation_position: {}".format(mutation_position))
            new_aa = result.group(3)
            #print("new_aa: {}".format(new_aa))
            #print("fullseq_list[mutation_position-1] : {}".format(fullseq_list[mutation_position-1] ))
            fullseq_list[mutation_position-1] = new_aa
    for content in content_list02:
        #print("="*10)
        #print("content: {}".format(content))
        pattern = r'^\s*([A-Z])(\d+)(.+)$'
        result = re.match(pattern, content) 
        gap_count = 0
        if result: #none以外の場合
            gap_position = int(result.group(2))
            #print("gap_position: {}".format(gap_position))
            del fullseq_list[gap_position-gap_count-1]
            gap_count += 1
    fullseq_mutation = "".join(fullseq_list)
    return fullseq_mutation

def test_data_gen_omicron_xec():
    # S	changes	T19I, R21T, T22N, A27S, S50L, F59S, V127F, G142D, F157S, R158G, L212I, V213G, L216F, H245N, A264D, I332V, G339H, K356T, S371F, S373P, S375F, T376A, R403K, D405N, R408S, K417N, N440K, V445H, G446S, N450D, L452W, L455S, F456L, N460K, S477N, T478K, N481K, E484K, F486P, Q493E, Q498R, N501Y, Y505H, E554K, A570V, D614G, P621S, H655Y, N679K, P681R, N764K, D796Y, S939F, Q954H, N969K, V1104L, P1143L
    # S	gaps	L24-, P25-, P26-, H69-, V70-, Y144-, N211-, V483-
    fullseq = "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQDVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEHVNNSYECDIPIGAGICASYQTQTNSPRRARSVASQSIIAYTMSLGAENSVAYSNNSIAIPTNFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQEVFAQVKQIYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDCLGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNQNAQALNTLVKQLSSNFGAISSVLNDILSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPAICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVSGNCDVVIGIVNNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDDSEPVLKGVKLHYT"
    fullseq_list = list(fullseq)
    df = pd.read_csv('./mutations/omicron_xec.csv', header=None)
    #print(df)
    #content = r'hellow python, 123,end' 
    content_list = df.iloc[0, 1:].tolist()
    #print(content_list)
    content_list02 = df.iloc[1, 1:9].tolist()
    #print(content_list02)
    for content in content_list:
        #print("="*10)
        #print("content: {}".format(content))
        pattern = r'^\s*([A-Z])(\d+)([A-Z])$'
        result = re.match(pattern, content)
        if result: #none以外の場合
            mutation_position = int(result.group(2))
            #print("metation_position: {}".format(mutation_position))
            new_aa = result.group(3)
            #print("new_aa: {}".format(new_aa))
            #print("fullseq_list[mutation_position-1] : {}".format(fullseq_list[mutation_position-1] ))
            fullseq_list[mutation_position-1] = new_aa
    for content in content_list02:
        #print("="*10)
        #print("content: {}".format(content))
        pattern = r'^\s*([A-Z])(\d+)(.+)$'
        result = re.match(pattern, content) 
        gap_count = 0
        if result: #none以外の場合
            gap_position = int(result.group(2))
            #print("gap_position: {}".format(gap_position))
            del fullseq_list[gap_position-gap_count-1]
            gap_count += 1
    fullseq_mutation = "".join(fullseq_list)
    return fullseq_mutation

if __name__ == '__main__':
    start = time.time()
    
    main()

    end = time.time()
    time_diff = end - start
    print(time_diff)
    print(time_diff/60)
    print(time_diff/60/60)
    print(time_diff/60/60/24)
    
    #dt_now = datetime.datetime.now()
    #print(dt_now)
    print("END.")

