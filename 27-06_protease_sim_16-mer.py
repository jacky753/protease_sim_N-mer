#protein-bert-seq_RF-regressor_ALL-Human-proteases_into_4-proteases_for_mutations_16-mer
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import csv

#import numpy as np
#import pandas as pd
from pandas import plotting 

import urllib.request 

#import matplotlib.pyplot as plt
#%matplotlib inline
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
#from torchvision import datasets, transforms
#import numpy as np
#import matplotlib.pyplot as plt
from tqdm import tqdm
import urllib.request
import sys
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
import os, glob
from sklearn.model_selection import KFold
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
#from pytorchtools import EarlyStopping
#early_stopping = EarlyStopping(patience=3, verbose=True)
from matplotlib import colormaps
list(colormaps)
#tirial about Optimizations
import torch.optim as optim
#import optuna
import json
import statistics
#from iTransformer import iTransformer
import math
#import numpy as np
#import pandas as pd
import seaborn as sns
#import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.nn import LayerNorm
from torch.utils.data import Dataset, DataLoader
from torch.nn import TransformerEncoder, TransformerDecoder, TransformerEncoderLayer, TransformerDecoderLayer
import datetime

#import calc_vari_001
import scipy
from scipy import signal

from urllib.request import urlopen
from lxml import etree

import itertools

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

from matplotlib.colors import ListedColormap
# import matplotlib.pyplot as plt
# import numpy as np

from sklearn.ensemble import RandomForestClassifier


from sklearn.tree import DecisionTreeClassifier
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

# データを読み込み
from sklearn.datasets import load_iris

from xgboost import XGBClassifier

# ランダムフォレスト回帰をインポート
from sklearn.ensemble import RandomForestRegressor
# 絶対平均誤差のインポート
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
    cur = con_db()
    merops_code_mece = create_merops_code_table()
    print("merops_code_mece.")
    #print(merops_code_mece)
    print(len(merops_code_mece))
    n_mer = 16
    
    #for k, protease in enumerate(merops_code_mece[124:]):
    #k_start=195
    #l = k_start
    #for k in range(k_start, len(merops_code_mece), 1):
    #for k, protease in enumerate(['S01.047', 'S01.021', 'S01.247', 'S01.034', 'S01.087']): 
    #for k, protease in enumerate(['S01.247']): 
    for k, protease in enumerate(['A01.003']): 
        #k = 2
        k = 100
        l = k

        print("="*10+"protease: {}".format(protease)+"_protease_turn: {}".format(k)+"="*10)

        merops_code = [protease]

        cur.execute("SELECT uniprot_acc, p1 FROM cleavage where code=(%s);", merops_code)
        subs = cur.fetchall()

        #if len(subs) > 0:
        #    os.makedirs("./outputdir_rf_reg_priority/{}_{}/".format(l, protease), exist_ok=True)
        #    os.makedirs("./PCA_test/outputdir_rf_reg_priority/{}_{}/".format(l, protease), exist_ok=True)
        #if len(subs) == 0:
        #    print("{} does not have the substrate data.".format(protease))
        #    continue
        
        print("="*10+"encoding data"+"="*10)
        pretrained_model_generator02, input_encoder02 = load_pretrained_model()
        protease_name = "{}".format(protease)
        #df_positive = pd.read_csv('./proteases_for_learning/cleave_pattern_one_letter_aa_S01.247.csv')
        #df_positive = pd.read_csv('./proteases/cleave_pattern_one_letter_aa_S01.247_update20241206.csv')
        df_positive = pd.read_csv('./proteases/cleave_pattern_one_letter_aa_{}.csv'.format(protease))

        seqs02= df_positive['cleave_pattern']

        print("seqs02.")
        print(seqs02)
        seq_len02= len(seqs02[0])+2
        print("seq_len")
        print(seq_len02)

        for i in range(len(seqs02)):
            #print("i in replace.")
            #print(i)
            #print("seqs.iloc[i]: {}".format(seqs.iloc[i]))
            #seqs.iloc[i].replace('-', '')
            seqs02.iloc[i] = seqs02.iloc[i].replace('-', '')
            #seqs02.iloc[i, 'cleave_pattern'] = seqs02.iloc[i, 'cleave_pattern'].str.replace('-', '')
        
        batch_size = 1
        model02 = get_model_with_hidden_layers_as_outputs(pretrained_model_generator02.create_model(seq_len02))
        print("model02.")
        print(model02)
        
        encoded_x_positive02 = input_encoder02.encode_X(seqs02, seq_len02)
        #print("encoded_x_positive.")
        #print(encoded_x_positive)
        #print(encoded_x_positive[0].shape)
        #print(encoded_x_positive[1].shape)
        #with open('encoded_x[1].csv', 'w') as f:
        #    writer = csv.writer(f)
        #    writer.writerow(encoded_x[1][0])
        local_representations_positive, global_representations_positive = model02.predict(encoded_x_positive02, batch_size=batch_size)
        print("local_representaions_positive02.")
        print(local_representations_positive.shape)



        pretrained_model_generator03, input_encoder03 = load_pretrained_model()
        #df_negative = pd.read_csv('./proteases_for_learning/negative_pattern_one_letter_aa_S01.247.csv')
        df_negative =  pd.read_csv('./proteases/negative_pattern_one_letter_aa_{}.csv'.format(protease))
        #df_negative = pd.read_csv('./proteases/cleave_pattern_one_letter_aa_A01.001.csv')

        seqs03= df_negative['negative_pattern']
        print("seqs03 on negative data.")
        print(seqs03)
        #seq_len= len(df_negative['negative_pattern'][0])+2
        seq_len03 = len(seqs03[0])+2
        print("seq_len03.")
        print(seq_len03)
        for i in range(len(seqs03)):
            seqs03.iloc[i] = seqs03.iloc[i].replace('-', '')
            #seqs03.lioc[, 'negative_pattern'] = seqs03['negative_pattern'].str.replace('-', '')
        for i in range(len(seqs03)):
            if len(seqs03[i]) > 8:
                print("There is outlier in the negative data.")
                print("i: {}".format(i))
                print(seqs03[i])
                break
            if i == len(seqs03)-1:
                print("There are no problem.")
        batch_size = 1
        model03 = get_model_with_hidden_layers_as_outputs(pretrained_model_generator03.create_model(seq_len03))
        print("model03.")
        print(model03)

        print("is it ok?")
        encoded_x_negative03 = input_encoder03.encode_X(seqs03, seq_len03)
        print("yes, ok!")
        
        print("seqs03 in negative.")
        print(seqs03)
        print(len(seqs03))
        print("seq_len03: {}".format(seq_len03))
        #print("encoded_x_negative.")
        #print(encoded_x_negative)
        #print(encoded_x_negative[0].shape)
        #print(encoded_x_negative[1].shape)
        #with open('encoded_x[1].csv', 'w') as f:
        #    writer = csv.writer(f)
        #    writer.writerow(encoded_x[1][0])
        print("Try model.predict() in negative data.")
        local_representations_negative, global_representations_negative = model03.predict(encoded_x_negative03, batch_size=batch_size)
        print("It's ok!")
        print("local_representations_negative.")
        print(local_representations_negative.shape)

        #df_negative = pd.read_csv('./proteases/cleave_pattern_one_letter_aa_A01.001.csv')

        #protease_name = "S01.034"
        #df_positive = pd.read_csv('./proteases/cleave_pattern_one_letter_aa_{}.csv'.format(protease_name))
        #df_negative = pd.read_csv('./proteases/negative_pattern_one_letter_aa_{}.csv'.format(protease_name))

        #protease_name = "S01.087"
        #df_positive = pd.read_csv('./proteases/cleave_pattern_one_letter_aa_A01.087_create.csv')
        #df_negative = pd.read_csv('./proteases/hogehoge.csv')


        #x_train = np.zeros([1, 20, 8, 1])
        cleave_pattern01 = df_positive.iloc[0, 3]#df_01.iloc[1, 3]
        uniprot_id01 = df_positive.iloc[0, 1]#df_01.iloc[1, 1]
        #print("cleave_pattern in df_01.")
        #print(cleave_pattern01)

        cleave_pattern01_pb = df_positive.iloc[0, 3]
        uniprot_id01_pb = df_positive.iloc[0, 1]


        #cleave_pattern02 = df_02.iloc[1, 3]
        #uniprot_id02 = df_02.iloc[1, 1]
        #print("cleave_pattern02 in df_02.")
        #print(cleave_pattern02)

        filename = "cleave_pattern_one_letter_aa_S01.247.csv"
        #print(f"filename: {filename}")


        filename_main = re.findall('(.*).csv$', filename)
        #print("filename_main.")
        #print(filename_main)
        save_filename = './learning_data_matrix/' + filename_main[0]
        #print(f"save_filename: {save_filename}")
        print('cleave_pattern01.')
        print(cleave_pattern01)
        df_test = char2vec(cleave_pattern01, n_mer)
        #print("df_test.")
        #print(df_test)

        test_array = df_test.to_numpy().flatten()
        df_concat = pd.DataFrame([test_array], index=['{}'.format(uniprot_id01+'+')])
        df_concat['label'] = 1
        #print("df_concat for test.")
        #print(df_concat)

        test_array_pb_seq = local_representations_positive[0].flatten()
        df_concat_pb_seq = pd.DataFrame([test_array_pb_seq], index=['{}'.format(uniprot_id01+'+')])
        df_concat_pb_seq['label'] = 1

        test_array_pb_go = global_representations_positive[0].flatten()
        df_concat_pb_go = pd.DataFrame([test_array_pb_go], index=['{}'.format(uniprot_id01+'+')])
        df_concat_pb_go['label'] = 1


        for i in range(len(df_positive)):
            #print("i: {}".format(i))
            uniprot_id = df_positive.iloc[i, 1]
            #print("uniprot_id: {}".format(uniprot_id))
            cleave_pattern = df_positive.iloc[i, 3]

            one_hot_seq = char2vec(cleave_pattern, n_mer)
            one_hot_array = one_hot_seq.to_numpy().flatten()
            pb_seq_array =  local_representations_positive[i].flatten()
            pb_go_array =  global_representations_positive[i].flatten()

            df = pd.DataFrame([one_hot_array], index=['{}'.format(uniprot_id)])
            df['label'] = 1
            df_pb_seq = pd.DataFrame([pb_seq_array], index=['{}'.format(uniprot_id)])
            df_pb_seq['label'] = 1
            df_pb_go = pd.DataFrame([pb_go_array], index=['{}'.format(uniprot_id)])
            df_pb_go['label'] = 1
            #print("df for df_concat.")
            #print(df)
            #df = pd.DataFrame([one_hot_seq], index=['{}'.format(uniprot_id)])
            #df['label'] = 1
            df_concat = pd.concat([df_concat, df])
            df_concat_pb_seq = pd.concat([df_concat_pb_seq, df_pb_seq])
            df_concat_pb_go = pd.concat([df_concat_pb_go, df_pb_go])

        print("df_concat.")
        #print(df_concat.drop(df.loc[0, :]))
        #print(df_concat.index) 
        df_concat = df_concat.drop(index='{}'.format(uniprot_id01+'+'))
        df_concat.to_csv('./PCA_test/outputdir_rf_reg_priority/{}_{}/df_concat.csv'.format(l, protease))
        #print(df_concat.index) 
        #print(df_concat)
        #print(df_concat.shape)

        df_concat_pb_seq = df_concat_pb_seq.drop(index='{}'.format(uniprot_id01+'+'))
        df_concat_pb_seq.to_csv('./PCA_test/outputdir_rf_reg_priority/{}_{}/df_concat_pb_seq.csv'.format(l, protease))
        print("df_concat_pb_seq.")
        print(df_concat_pb_seq.index) 
        print(df_concat_pb_seq)
        print(df_concat_pb_seq.shape)

        df_concat_negative = pd.DataFrame([test_array], index=['{}'.format(uniprot_id01+'+')])
        df_concat_negative['label'] = 0 
        #print("df_concat_negative.")
        #print(df_concat_negative)
        #print(df_concat_negative.shape)

        test_array_pb_seq = local_representations_negative[0].flatten()
        df_concat_pb_seq_negative = pd.DataFrame([test_array_pb_seq], index=['{}'.format(uniprot_id01+'+')])
        df_concat_pb_seq_negative['label'] = 0

        test_array_pb_go = global_representations_negative[0].flatten()
        df_concat_pb_go_negative = pd.DataFrame([test_array_pb_go], index=['{}'.format(uniprot_id01+'+')])
        df_concat_pb_go_negative['label'] = 0

        for i in range(len(df_negative)):
            #print("i in negative.: {}".format(i))
            #print(df_negative)
            uniprot_id = df_negative.iloc[i, 1]
            #uniprot_id = df_negative['uniprot_id'][i]
            print("uniprot_id: {}".format(uniprot_id))
            cleave_pattern = df_negative.iloc[i, 3]

            #cleave_pattern = df_negative.iloc['cleave_pattern'][i]
            print("cleave_pattern for df_concat_negative.: {}".format(cleave_pattern))
            one_hot_seq = char2vec(cleave_pattern, n_mer)
            one_hot_array = one_hot_seq.to_numpy().flatten()
            pb_seq_array =  local_representations_negative[i].flatten()
            pb_go_array =  global_representations_negative[i].flatten()
            
            df = pd.DataFrame([one_hot_array], index=['{}'.format(uniprot_id)])
            df['label'] = 0   
            df_pb_seq = pd.DataFrame([pb_seq_array], index=['{}'.format(uniprot_id)])
            df_pb_seq['label'] = 0
            df_pb_go = pd.DataFrame([pb_go_array], index=['{}'.format(uniprot_id)])
            df_pb_go['label'] = 0

            #print("df for df_concat.")
            #print(df)
            #df = pd.DataFrame([one_hot_seq], index=['{}'.format(uniprot_id)])
            #df['label'] = 1
            df_concat_pb_seq_negative = pd.concat([df_concat_pb_seq_negative, df_pb_seq])
            df_concat_pb_go_negative = pd.concat([df_concat_pb_go_negative, df_pb_go])
            #print("df for df_concat.")
            #print(df)
            df_concat_negative = pd.concat([df_concat_negative, df])
        
        
        df_concat_negative = df_concat_negative.drop(index='{}'.format(uniprot_id01+'+'))
        df_concat_negative.to_csv('./PCA_test/outputdir_rf_reg_priority/{}_{}/df_concat_negative.csv'.format(l, protease))
        #print("df_concat_negative completed.")
        #print(df_concat_negative)
        #print(df_concat_negative.shape)

        df_concat_pb_seq_negative = df_concat_pb_seq_negative.drop(index='{}'.format(uniprot_id01+'+'))
        df_concat_pb_seq_negative.to_csv('./PCA_test/outputdir_rf_reg_priority/{}_{}/df_concat_pb_seq_negative.csv'.format(l, protease))
        print("df_concat_pb_seq_negative.")
        print(df_concat_pb_seq_negative.index) 
        print(df_concat_pb_seq_negative)
        print(df_concat_pb_seq_negative.shape)

        df_concat_pb_go_negative = df_concat_pb_go_negative.drop(index='{}'.format(uniprot_id01+'+'))
        df_concat_pb_go_negative.to_csv('./PCA_test/outputdir_rf_reg_priority/{}_{}/df_concat_pb_go_negative.csv'.format(l, protease))
        print("df_concat_pb_go_negative.")
        print(df_concat_pb_go_negative.index) 
        print(df_concat_pb_go_negative)
        print(df_concat_pb_go_negative.shape)

        pca_return = pca_plotting(df_concat_pb_seq, df_concat_pb_seq_negative, protease_name, l, protease, df_positive)

        device = set_device()
        #temp = set_data_param()
        #print("df_learning may be label.")
        data_columns_positive = len(df_concat_pb_seq.columns)-1
        print("data_columns_positive.")
        print(data_columns_positive)
        #print(df_learning.iloc[:, data_columns])
        x_data_positive = df_concat_pb_seq.iloc[:, 0:data_columns_positive]
        print("x_data_positive.")
        print(x_data_positive)
        y_data_positive = df_concat_pb_seq.iloc[:, data_columns_positive]
        print("y_data_positive.")
        print(y_data_positive)
        len_y_data_positive = len(y_data_positive)


        if len_y_data_positive // 5 == 0:
            five_cv_posi = -1
            print("five_cv_posi.")
            print(five_cv_posi)
            x_data_tr_posi = x_data_positive.iloc[ :five_cv_posi, :].to_numpy()
            print("x_data_tr_posi.")
            print(x_data_tr_posi)
            y_data_tr_posi = y_data_positive.iloc[ :five_cv_posi].to_numpy()
            print("y_data_tr_posi.")
            print(y_data_tr_posi)
            x_data_val_posi = x_data_positive.iloc[five_cv_posi:, :].to_numpy()
            y_data_val_posi = y_data_positive.iloc[five_cv_posi: ].to_numpy()
        else:
            five_cv_posi = len_y_data_positive // 5
            print("five_cv_posi*4.")
            print(five_cv_posi*4)
            x_data_tr_posi = x_data_positive.iloc[ :five_cv_posi*4, :].to_numpy()
            y_data_tr_posi = y_data_positive.iloc[ :five_cv_posi*4].to_numpy()
            x_data_val_posi = x_data_positive.iloc[five_cv_posi*4: , :].to_numpy()
            y_data_val_posi = y_data_positive.iloc[five_cv_posi*4: ].to_numpy()

        data_columns_negative = len(df_concat_pb_seq_negative.columns)-1
        #print(df_learning.iloc[:, data_columns])
        x_data_negative = df_concat_pb_seq_negative.iloc[:, 0:data_columns_negative]
        print("x_data_negative.")
        print(x_data_negative)
        y_data_negative = df_concat_pb_seq_negative.iloc[:, data_columns_negative]
        print("y_data_negative.")
        print(y_data_negative)

        len_y_data_negative = len(y_data_negative)
        if len_y_data_negative // 5 == 0:
            five_cv_nega = -1
            x_data_tr_nega = x_data_negative.iloc[ :five_cv_nega, :].to_numpy()
            print("x_data_tr_nega.")
            print(x_data_tr_nega)
            y_data_tr_nega = y_data_negative.iloc[ :five_cv_nega].to_numpy()
            print("y_data_tr_nega.")
            print(y_data_tr_nega)
            x_data_val_nega = x_data_negative.iloc[five_cv_nega: , :].to_numpy()
            y_data_val_nega = y_data_negative.iloc[five_cv_nega: ].to_numpy()
        else:
            five_cv_nega = len_y_data_positive // 5
            x_data_tr_nega = x_data_negative.iloc[ :five_cv_nega*4, :].to_numpy()
            y_data_tr_nega = y_data_negative.iloc[ :five_cv_nega*4].to_numpy()
            x_data_val_nega = x_data_negative.iloc[five_cv_nega*4: , :].to_numpy()
            y_data_val_nega = y_data_negative.iloc[five_cv_nega*4: ].to_numpy()

        x_train = np.concatenate([x_data_tr_posi, x_data_tr_nega], axis=0)
        y_train = np.concatenate([y_data_tr_posi, y_data_tr_nega], axis=0)

        x_val = np.concatenate([x_data_val_posi, x_data_val_nega], axis=0)
        y_val = np.concatenate([y_data_val_posi, y_data_val_nega], axis=0)
        
        x_data = np.concatenate([x_data_positive, x_data_negative], axis=0)
        y_data = np.concatenate([y_data_positive, y_data_negative], axis=0)

        x_combined = np.vstack((x_train, x_val))
        y_combined = np.hstack((y_train, y_val))

        print("x_train.")
        print(x_train)
        print(x_train.shape)
        print("y_train.")
        print(y_train)
        print(y_train.shape)

        print("x_val")
        print(x_val)
        print(x_val.shape)
        print("y_data_val")
        print(y_val)
        print(y_val.shape)



        forest = RandomForestRegressor()
        if len(df_positive) == 1 or len(df_positive) == 2:
            n_splits = 2
        else:
            n_splits = 3
            stratifiedkfold = StratifiedKFold(n_splits=n_splits)
            cvs = cross_val_score(forest, x_data, y_data, cv=stratifiedkfold)
            cvs_ave = sum(cvs)/len(cvs)
            #df_cvs = pd.DataFrame(cvs, columns=["1", "2", "3"], index=["Cross-validation scores"])
            df_cvs = pd.DataFrame(cvs, columns=["Cross-validation scores"], index=["1", "2", "3"])
            df_cvs_ave = pd.DataFrame(cvs_ave, columns=["cvs_ave"], index=["0"])
            df_cvs.to_csv("./outputdir_rf_reg_priority/{}_{}/cross-validation_scores_{}.csv".format(l, protease, protease_name))
            df_cvs_ave.to_csv("./outputdir_rf_reg_priority/{}_{}/cvs_average_{}.csv".format(l, protease, protease_name))
            print('Cross-validation scores: \n{}'.format(cvs))



        #forest = RandomForestClassifier(criterion='entropy',
        #                            n_estimators=10, random_state=1, n_jobs=2)
        forest = RandomForestRegressor()
        if len(df_positive) == 1 or len(df_positive) == 2:
            forest.fit(x_data, y_data)
        else:
            forest.fit(x_train, y_train)
        print(forest.feature_importances_)
        df_feature_importances = pd.DataFrame(forest.feature_importances_, 
                                                columns=["feature_importances"], 
                                                index=list(range(len(forest.feature_importances_))))
        df_feature_importances.to_csv("./outputdir_rf_reg_priority/{}_{}/df_feature_importances_{}.csv".format(l, protease, protease_name))
        predicteds = forest.predict(x_val)
        print(predicteds)
        print(type(predicteds))
        print(len(predicteds))
        print(predicteds.shape)
        df_y_val = pd.DataFrame(y_val, columns=["y_val"], index=list(range(len(y_val))))
        df_val_predicteds = pd.DataFrame(predicteds, columns=["predicteds_val"], index=list(range(len(predicteds))))
        #df_val_predicteds = pd.DataFrame([y_val, predicteds], columns=["y_val", "val_predict"], index=list(range(len(predicteds))))
        df_val_predicteds = pd.concat([df_y_val, df_val_predicteds], axis=1)
        df_val_predicteds.to_csv("./outputdir_rf_reg_priority/{}_{}/df_val_predicteds_{}.csv".format(l, protease, protease_name))
        
        # 精度検証
        mae = mean_absolute_error(y_val, predicteds)
        print("mae: {}".format(mae))
        mse = mean_squared_error(y_val, predicteds)
        print("mse: {}".format(mse))
        df_val_error = pd.DataFrame([[mae, mse]], columns=["val_mae", "val_mse"], index=["0"])
        #df_val_error = pd.DataFrame([[mae], [mse]], columns=["0"], index=["val_mae", "val_mse"])
        df_val_error.to_csv("./outputdir_rf_reg_priority/{}_{}/df_val_error_{}.csv".format(l, protease, protease_name))

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
        for i in range(12):
            if i == 0:
                fullseq = origin_fullseq
                mutation_name = "origin"
            elif i == 1:
                fullseq = test_data_gen_alpha()
                mutation_name = "alpha"
            elif i == 2:
                fullseq = test_data_gen_beta()
                mutation_name = "beta"
            elif i == 3:
                fullseq = test_data_gen_gamma()
                mutation_name = "gamma"
            elif i == 4:
                fullseq = test_data_gen_delta()
                mutation_name = "delta"
            elif i == 5:
                fullseq = test_data_gen_omicron_ba1()
                mutation_name = "omicron_ba1"
            elif i == 6:
                fullseq = test_data_gen_omicron_ba2()
                mutation_name = "omicron_ba2"
            elif i == 7:
                fullseq = test_data_gen_omicron_ba3()
                mutation_name = "omicron_ba3"
            elif i == 8:
                fullseq = test_data_gen_omicron_jn1()
                mutation_name = "omicron_jn1"
            elif i == 9:
                fullseq = test_data_gen_omicron_eg5()
                mutation_name = "omicron_eg5"
            elif i == 10:
                fullseq = test_data_gen_omicron_kp3()
                mutation_name = "omicron_kp3"
            elif i == 11:
                fullseq = test_data_gen_omicron_xec()
                mutation_name = "omicron_xec"
            else:
                pass

            spike_protein_eight_aa_list = test_data_gen(fullseq)
            #print("spike_protein_eight_aa_list.")
            #rint(spike_protein_eight_aa_list)
            #print(len(spike_protein_eight_aa_list))
            
            #x_test = "PSKRSFIE"
            #x_test02 = "VGEIPVAY"
            #seqs= df_01['cleave_pattern']
            #df_test_pattern = pd.DataFrame([x_test, x_test02], columns=["test_pattern"], index=["0", "1"])
            df_test_pattern = pd.DataFrame(spike_protein_eight_aa_list, columns=["test_pattern"], index=list(range(len(spike_protein_eight_aa_list))))
            seqs = df_test_pattern["test_pattern"]
            #print("seqs on df_test_pattern.")
            #print(seqs)
            #print(len(seqs))
            #j=20
            #for i in range(j):
            #   print("i: {}".format(615-1-5+i))
            #    print(seqs[615-1-int(j/2)+i])
            seq_len= len(seqs[0])+2

            #print("seq_len")
            #print(seq_len)
            batch_size = 1
            model = get_model_with_hidden_layers_as_outputs(pretrained_model_generator.create_model(seq_len))
            #print("model.")
            #print(model)
            encoded_x = input_encoder.encode_X(seqs, seq_len)
            #print("encoded_x.")
            #print(encoded_x)
            #print(len(encoded_x))
            local_representations, global_representations = model.predict(encoded_x, batch_size=batch_size)
            #print("local_representations[0] on test_data.")
            #print(local_representations)
            #rint(local_representations.shape)
            #rint(len(local_representations))
            #print(local_representations[0])
            #rint(local_representations.shape)
            #rint(len(local_representations[0][0]))
            #rint(local_representations.shape[2])



            #x_test = local_representations[0].flatten()
            #x_test02 = local_representations[1].flatten()
            #x_test = local_representations.reshape(-1, seq_len*len(local_representations.shape[2]))
            x_test = local_representations.reshape(-1, int(local_representations.shape[1]*local_representations.shape[2]))
            
            
            #x_test =  np.expand_dims(x_test, axis=0)
            #x_test03 = np.concatenate([[x_test], [x_test02]], axis=0)
            #print("x_test before predict.")
            #print(x_test)
            #print(type(x_test03))
            #print(x_test.shape)
            #y_test = np.array([1.0, 0.0]).reshape(-1, 1)
            #y_test = np.zeros(len(x_test), 1)
            y_test = np.zeros((len(x_test), 1))
            #print("y_test.")
            #print(y_test)
            #print(y_test.shape)
            #print("x_test[615].")
            #print(x_test[615-1])
            #y_test = np.put(y_test, [615-1], 1.0)
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
            print(predictdata)
            print(predictdata.shape)
            df_predict = pd.DataFrame(predictdata, columns=["predict"], index=list(range(len(predictdata))))
            df_predict.to_csv("./outputdir_rf_reg_priority/{}_{}/predict_sprotein_{}.csv".format(l, protease, protease_name))
            df_concat_predict_seqs = pd.concat([df_predict, seqs], axis=1)
            df_concat_predict_seqs.to_csv("./outputdir_rf_reg_priority/{}_{}/predict_sprotein_seqs_{}_{}.csv".format(l, protease, protease_name, mutation_name))
        print("END.")
        l += 1
        ######df_learning = pd.concat([df_concat_pb_seq, df_concat_pb_seq_negative])
        #print("END.")
def con_db():
    # コネクションの作成
    conn = mydb.connect(
        host='localhost',
        port=3306,
        user='hoge',
        password='foo',
        database='meropsweb12_1'
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

    print("columns_list.")
    print(columns_list)
    df = pd.DataFrame(arr,
                    #columns = ["P4","P3","P2", "P1", "P1dash", "P2dash", "P3dash", "P4dash"],
                    columns = columns_list,
                    index = ["A","C","D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"])
    for j in range(len(cleave_pattern)): 
        #print("j in char2vec func.: {}".format(j))
        aa = cleave_pattern[j]  
        if aa == "A":
            #x_train[num][0][j] += 1.0
            df.iloc[0, j] +=1.0
        elif aa == "C":
            #x_train[num][1][j] += 1.0 
            df.iloc[1, j] +=1.0
        elif aa == "D":
            #x_train[num][2][j] += 1.0 
            df.iloc[2, j] +=1.0
        elif aa == "E":
            #x_train[num][3][j] += 1.0 
            df.iloc[3, j] +=1.0
        elif aa == "F":
            #x_train[num][4][j] += 1.0
            df.iloc[4, j] +=1.0
        elif aa == "G":
            #x_train[num][5][j] += 1.0
            df.iloc[5, j] +=1.0
        elif aa == "H":
            #x_train[num][6][j] += 1.0
            df.iloc[6, j] +=1.0
        elif aa == "I":
            #x_train[num][7][j] += 1.0
            df.iloc[7, j] +=1.0
        elif aa == "K":
            #x_train[num][8][j] += 1.0
            df.iloc[8, j] +=1.0
        elif aa == "L":
            #x_train[num][9][j] += 1.0
            df.iloc[9, j] +=1.0
        elif aa == "M":
            #x_train[num][10][j] += 1.0
            df.iloc[10, j] +=1.0
        elif aa == "N":
            #x_train[num][11][j] += 1.0
            df.iloc[11, j] +=1.0
        elif aa == "P":
            #x_train[num][12][j] += 1.0
            df.iloc[12, j] +=1.0
        elif aa == "Q":
            #x_train[num][13][j] += 1.0
            df.iloc[13, j] +=1.0
        elif aa == "R":
            #x_train[num][14][j] += 1.0
            df.iloc[14, j] +=1.0
        elif aa == "S":
            #x_train[num][15][j] += 1.0
            df.iloc[15, j] +=1.0
        elif aa == "T":
            #x_train[num][16][j] += 1.0
            df.iloc[16, j] +=1.0
        elif aa == "V":
            #x_train[num][17][j] += 1.0
            df.iloc[17, j] +=1.0
        elif aa == "W":
            #x_train[num][18][j] += 1.0
            df.iloc[18, j] +=1.0
        elif aa == "Y":
            #x_train[num][19][j] += 1.0
            df.iloc[19, j] +=1.0
        else:
            pass
    return df
    


# learning data.
def pca_plotting(df_concat_pb_seq, df_concat_pb_seq_negative, protease_name, l, protease, df_positive):
    #print("="*10+"learning data: pb seq."+"="*10)
    print("="*10+"start pca plotting."+"="*10)
    df_learning = pd.concat([df_concat_pb_seq, df_concat_pb_seq_negative])
    #print(df_learning)
    #print(df_learning.shape)
    df_learning.to_csv('./PCA_test/outputdir_rf_reg_priority/{}_{}/df_learning_pb_seq.csv'.format(l, protease))
    #print("df_learning.")
    #print(df_learning)
    #print(df_learning.shape)

    pca_learning = PCA()
    #pca.fit(dfs)
    #print("df_learning may be label.")
    data_columns = len(df_learning.columns)-1
    #print(df_learning.iloc[:, data_columns])


    x_data = df_learning.iloc[:, 0:data_columns]
    pc_num = min(len(x_data.index), len(x_data.columns))
    print("pc_num on learning data.: {}".format(pc_num))

    #plotting.scatter_matrix(x_data, figsize=(8, 8), c=list(df_learning.iloc[:, 64]), alpha=0.5)
    #plt.show()
    x_data.columns = x_data.columns.astype(str)

    pca_learning.fit(x_data)
    feature_learning = pca_learning.transform(x_data)
    #print("feature.")
    #print(feature_learning)
    #print(type(feature_learning))
    #print(feature_learning.shape)
    #rint(feature_learning.shape[1])

    df_learning_pca = pd.DataFrame(feature_learning, columns=["PC{}".format(x + 1) for x in range(pc_num)])
    #print("="*10+"df_learning_pca."+"="*10)
    #print(df_learning_pca)
    df_learning_pca.to_csv('./PCA_test/outputdir_rf_reg_priority/{}_{}/df_learning_pca_feature.csv'.format(l, protease))

    # 第一主成分と第二主成分でプロットする
    fig, ax = plt.subplots()
    ax.grid()
    #plt.xlabel("PC1")
    #plt.ylabel("PC2")
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title("learning pca: {}".format(protease_name))
    plt.figure(figsize=(6, 6))
    #plt.scatter(feature[:, 0], feature[:, 1], alpha=0.8, c=list(df_concat.iloc[:, 0]))
    ax.scatter(feature_learning[:, 0], feature_learning[:, 1], alpha=0.8, c=list(df_learning.iloc[:, data_columns]))
    fig.tight_layout()
    fig.savefig('./PCA_test/outputdir_rf_reg_priority/{}_{}/learning_pca_plot_{}.png'.format(l, protease, protease_name))
    #plt.show()

    #plotting.scatter_matrix(pd.DataFrame(feature_learning, 
    #                        columns=["PC{}".format(x + 1) for x in range(pc_num)]), 
    #                        figsize=(8, 8), c=list(df_learning.iloc[:, 64]), alpha=0.5) 
    #plt.show()

    learning_df_explained_vr = pd.DataFrame(pca_learning.explained_variance_ratio_, index=["PC{}".format(x + 1) for x in range(pc_num)])
    print("="*10+"learning_df_explained_vr"+"="*10)
    print(learning_df_explained_vr)
    learning_df_explained_vr.to_csv('./PCA_test/outputdir_rf_reg_priority/{}_{}/learning_df_explained_variance_ratio.csv'.format(l, protease))

    fig, ax = plt.subplots()
    ax.grid()
    ax.set_title("Cumulative contribution rate in learning dataset: {}".format(protease_name))
    plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
    #plt.plot([0] + list( np.cumsum(pca.explained_variance_ratio_)), "-o")
    ax.plot([0] + list( np.cumsum(pca_learning.explained_variance_ratio_)), "-o")
    #plt.xlabel("Number of principal components")
    #plt.ylabel("Cumulative contribution rate")
    ax.set_xlabel('Number of principal components')
    ax.set_ylabel('Cumulative contribution rate')
    fig.tight_layout()
    fig.savefig('./PCA_test/outputdir_rf_reg_priority/{}_{}/cumulative_contribution_rate_in-learning-data_{}.png'.format(l, protease, protease_name))
    #plt.show()


    learning_df_eigenvalue = pd.DataFrame(pca_learning.explained_variance_, index=["PC{}".format(x + 1) for x in range(pc_num)])
    print("="*10+"df_eigenvalue."+"="*10)
    print(learning_df_eigenvalue)
    learning_df_eigenvalue.to_csv('./PCA_test/outputdir_rf_reg_priority/learning_df_eigenvalue.csv')

    learning_df_eigenvector = pd.DataFrame(pca_learning.components_, columns=x_data.columns, index=["PC{}".format(x + 1) for x in range(pc_num)])
    print("="*10+"df_eigenvector"+"="*10)
    print(learning_df_eigenvector)
    learning_df_eigenvector.to_csv('./PCA_test/outputdir_rf_reg_priority/{}_{}/learning_df_eigenvector.csv'.format(l, protease))

    # 第一主成分と第二主成分における観測変数の寄与度をプロットする
    fig, ax = plt.subplots()
    plt.figure(figsize=(6, 6))
    ax.grid()
    #plt.xlabel("PC1")
    #plt.ylabel("PC2")
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title("explanatory variables contribution map on learning data: {}".format(protease_name))
    for x, y, name in zip(pca_learning.components_[0], pca_learning.components_[1], df_learning.columns[:data_columns]):
        #plt.text(x, y, name)
        ax.text(x, y, name)
    #plt.scatter(pca.components_[0], pca.components_[1], alpha=0.8)
    ax.scatter(pca_learning.components_[0], pca_learning.components_[1], alpha=0.8)
    fig.tight_layout()
    fig.savefig('./PCA_test/outputdir_rf_reg_priority/{}_{}/components_contribution_map_on-learning-data_{}.png'.format(l, protease, protease_name))
    ##plt.show()


    #==================================================================================================================
    fig, ax = plt.subplots()
    ax.grid()
    ax = plt.gca()  # gca stands for 'get current axis'
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data',0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data',0))
    #plt.xlabel("PC1")
    #plt.ylabel("PC2")
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title("learning pca with color-coding each labels.: {}".format(protease_name))
    #ax.spines['bottom'].set_position(('data', 0))
    #ax.spines['left'].set_position(('data', 0))
    plt.figure(figsize=(6, 6))
    #plt.scatter(feature[:, 0], feature[:, 1], alpha=0.8, c=list(df_concat.iloc[:, 0]))
    #ax.scatter(feature_learning[:, 0], feature_learning[:, 1], alpha=0.8, c=list(df_learning.iloc[:, 0]))
    for i in range(len(df_learning)):
        if df_learning.iloc[i, data_columns] == 1:
            #print(i)
            #print("positive.")
            #print(df_learning.iloc[i, :])
            #print(feature_learning[i, 0])
            #print(feature_learning[i, 1])
            ax.scatter(feature_learning[i, 0], feature_learning[i, 1], alpha=0.8, c="blue")
        else:
            #print(i)
            #print("negative.")
            #print(df_learning.iloc[i, :])
            #print(feature_learning[i, 0])
            #print(feature_learning[i, 1])
            ax.scatter(feature_learning[i, 0], feature_learning[i, 1], alpha=0.8, c="red")

    fig.tight_layout()
    fig.savefig('./PCA_test/outputdir_rf_reg_priority/{}_{}/learning_pca_plot_with-color-coding_{}.png'.format(l, protease, protease_name))
    #plt.show()

    #===========================================================================================================
    if len(df_positive) == 1:
        return -1
    else:
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': '3d'})
        ax.grid()
        ax = plt.gca()  # gca stands for 'get current axis'
        #ax.spines['right'].set_color('none')
        #ax.spines['top'].set_color('none')
        #ax.xaxis.set_ticks_position('bottom')
        #ax.spines['bottom'].set_position(('data',0))
        #ax.yaxis.set_ticks_position('left')
        #ax.spines['left'].set_position(('data',0))
        #plt.xlabel("PC1")
        #plt.ylabel("PC2")
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.set_title("learning pca with color-coding each labels.: {}".format(protease_name))
        #ax.spines['bottom'].set_position(('data', 0))
        #ax.spines['left'].set_position(('data', 0))
        plt.figure(figsize=(6, 6))
        for i in range(len(df_learning)):
            if df_learning.iloc[i, data_columns] == 1:
                #print(i)
                #print("positive.")
                #print(df_learning.iloc[i, :])
                #print(feature_learning[i, 0])
                #print(feature_learning[i, 1])
                ax.scatter(feature_learning[i, 0], feature_learning[i, 1], feature_learning[i, 2], alpha=0.8, c="blue", marker="v")
            else:
                #print(i)
                #print("negative.")
                #print(df_learning.iloc[i, :])
                #print(feature_learning[i, 0])
                #print(feature_learning[i, 1])
                ax.scatter(feature_learning[i, 0], feature_learning[i, 1], feature_learning[i, 2], alpha=0.8, c="red", marker="o")

        fig.tight_layout()
        fig.savefig('./PCA_test/outputdir_rf_reg_priority/{}_{}/learning_pca_3d-scatter_with-color-coding_{}.png'.format(l, protease, protease_name))
        #plt.show()
    return 0




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

def test_data_gen(fullseq):
    #uniprot_id = "P0DTC2"
    #fullseq = aaseq_from_uid(uniprot_id, 0, 0)
    # 1273 aa
    # https://www.ncbi.nlm.nih.gov/nuccore/NC_045512.2
    #fullseq = "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQDVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEHVNNSYECDIPIGAGICASYQTQTNSPRRARSVASQSIIAYTMSLGAENSVAYSNNSIAIPTNFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQEVFAQVKQIYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDCLGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNQNAQALNTLVKQLSSNFGAISSVLNDILSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPAICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVSGNCDVVIGIVNNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDDSEPVLKGVKLHYT"
    #fullseq_list = list(fullseq)


    spike_protein_aa_list = []

    for loc_ in range(len(fullseq)-7):
        eight_seq = fullseq[loc_: loc_+8]
        spike_protein_aa_list.append(eight_seq)

    return spike_protein_aa_list

def test_data_gen_alpha():
    # S	changes	N501Y, A570D, D614G, P681H, T716I, S982A, D1118H
    # S	gaps	H69-, V70-, Y144-
    fullseq = "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQDVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEHVNNSYECDIPIGAGICASYQTQTNSPRRARSVASQSIIAYTMSLGAENSVAYSNNSIAIPTNFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQEVFAQVKQIYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDCLGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNQNAQALNTLVKQLSSNFGAISSVLNDILSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPAICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVSGNCDVVIGIVNNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDDSEPVLKGVKLHYT"
    print("aa685")
    print(fullseq[685-1 -3 -3: 685-1 +5])
    print(fullseq[685-1 -3 -3: 685-1 +5])
    print("aa815")
    print(fullseq[815-1 -3: 815-1 +5])
    fullseq_list = list(fullseq)
    print(fullseq_list[501-1])
    print(fullseq_list[570-1])
    print(fullseq_list[614-1])
    print(fullseq_list[681-1])
    print(fullseq_list[716-1])
    print(fullseq_list[982-1])
    print(fullseq_list[1118-1])
    print("N, A, D, P, T, S, D.")
    print(fullseq_list[69-1])
    print(fullseq_list[70-1])
    print(fullseq_list[144-1])
    print("H, V, Y.")
    fullseq_list[501-1] = "Y"
    fullseq_list[570-1] = "D"
    fullseq_list[614-1] = "G"
    fullseq_list[681-1] = "H"
    fullseq_list[716-1] = "I"
    fullseq_list[982-1] = "A"
    fullseq_list[1118-1] = "H"
    print("H69-: {}".format(fullseq_list[69-1]))
    del fullseq_list[69-1] 
    print("V70-: {}".format(fullseq_list[70-1-1]))
    del fullseq_list[70-1-1]
    print("Y144-: {}".format(fullseq_list[144-2-1]))
    del fullseq_list[144-2-1]
    fullseq_mutation = "".join(fullseq_list)

    return fullseq_mutation

def test_data_gen_beta():
    # S	changes	D80A, D215G, K417N, E484K, N501Y, D614G, A701V
    # S	gaps	L241-, L242-, A243-
    fullseq = "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQDVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEHVNNSYECDIPIGAGICASYQTQTNSPRRARSVASQSIIAYTMSLGAENSVAYSNNSIAIPTNFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQEVFAQVKQIYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDCLGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNQNAQALNTLVKQLSSNFGAISSVLNDILSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPAICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVSGNCDVVIGIVNNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDDSEPVLKGVKLHYT"
    fullseq_list = list(fullseq)
    fullseq_list[80-1] = "A"
    fullseq_list[215-1] = "G"
    fullseq_list[417-1] = "N"
    fullseq_list[484-1] = "K"
    fullseq_list[501-1] = "Y"
    fullseq_list[614-1] = "G"
    fullseq_list[701-1] = "V"
    del fullseq_list[241 -1]
    del fullseq_list[242-1 -1]
    del fullseq_list[243-1-1 -1]
    print(fullseq_list[501-3 -1])
    fullseq_mutation = "".join(fullseq_list)

    return fullseq_mutation

def test_data_gen_gamma():
    # S	changes	L18F, T20N, P26S, D138Y, R190S, K417T, E484K, N501Y, D614G, H655Y, T1027I, V1176F
    fullseq = "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQDVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEHVNNSYECDIPIGAGICASYQTQTNSPRRARSVASQSIIAYTMSLGAENSVAYSNNSIAIPTNFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQEVFAQVKQIYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDCLGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNQNAQALNTLVKQLSSNFGAISSVLNDILSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPAICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVSGNCDVVIGIVNNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDDSEPVLKGVKLHYT"
    fullseq_list = list(fullseq)
    fullseq_list[18-1] = "F"
    fullseq_list[20-1] = "N"
    fullseq_list[26-1] = "S"
    fullseq_list[138-1] = "Y"
    fullseq_list[190-1] = "S"
    fullseq_list[417-1] = "T"
    fullseq_list[484-1] = "K"
    fullseq_list[501-1] = "Y"
    fullseq_list[614-1] = "G"
    fullseq_list[655-1] = "Y"
    fullseq_list[1027-1] = "I"
    fullseq_list[1176-1] = "F"

    #del fullseq_list[241 -1]
    #del fullseq_list[242-1 -1]
    #del fullseq_list[243-1-1 -1]
    print("fullseq_list[501-3 -1] in gen_gamma().")
    print(fullseq_list[501-3 -1])
    fullseq_mutation = "".join(fullseq_list)
    return fullseq_mutation

def test_data_gen_delta():
    fullseq = "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQDVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEHVNNSYECDIPIGAGICASYQTQTNSPRRARSVASQSIIAYTMSLGAENSVAYSNNSIAIPTNFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQEVFAQVKQIYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDCLGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNQNAQALNTLVKQLSSNFGAISSVLNDILSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPAICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVSGNCDVVIGIVNNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDDSEPVLKGVKLHYT"
    fullseq_list = list(fullseq)
    # S	changes	T19R, G142D, R158G, L452R, T478K, D614G, P681R, D950N
    # S	gaps	E156-, F157-
    #s_changes_list = [T19R, G142D, R158G, L452R, T478K, D614G, P681R, D950N]
    df = pd.read_csv('./mutations/delta.csv', header=None)
    print(df)
    #content = r'hellow python, 123,end' 
    content_list = df.iloc[0, :].tolist()
    print(content_list)
    # ()で取りたい文字を
    content_list02 = df.iloc[1, :].tolist()
    content_list02 = content_list02[:2]
    print(content_list02)
    for content in content_list:
        print("="*10)
        print("content: {}".format(content))
        pattern = r'^\s*([A-Z])(\d+)([A-Z])$'

        result = re.match(pattern, content)

        if result: #none以外の場合
            # group()で全文字を
            #print(result.group(0))  # hellow python, 123,end
            # group(1)で数字を
            
            #print(result.group(1)) # 123
            #print(result.group(2))
            #rint(result.group(3))
            mutation_position = int(result.group(2))
            print("metation_position: {}".format(mutation_position))
            new_aa = result.group(3)
            print("new_aa: {}".format(new_aa))
            print("fullseq_list[mutation_position-1] : {}".format(fullseq_list[mutation_position-1] ))
            fullseq_list[mutation_position-1] = new_aa
    for content in content_list02:
        print("="*10)

        print("content: {}".format(content))
        #if not(type(content) == 'str'):
        #    pass
        pattern = r'^\s*([A-Z])(\d+)(.+)$'
        result = re.match(pattern, content) 
        gap_count = 0
        if result: #none以外の場合
            # group()で全文字を
            #print(result.group(0))  # hellow python, 123,end
            # group(1)で数字を
            
            #print(result.group(1)) # 123
            #print(result.group(2))
            #rint(result.group(3))
            gap_position = int(result.group(2))
            print("gap_position: {}".format(gap_position))
            #new_aa = result.group(3)
            #fullseq_list[mutation_position-1] = new_aa
            del fullseq_list[gap_position-gap_count-1]
            gap_count += 1
    fullseq_mutation = "".join(fullseq_list)
    return fullseq_mutation

def test_data_gen_omicron_ba1():
    # S	changes	A67V, T95I, Y145D, L212I, G339D, S371L, S373P, S375F, K417N, N440K, G446S, S477N, T478K, E484A, Q493R, G496S, Q498R, N501Y, Y505H, T547K, D614G, H655Y, N679K, P681H, N764K, D796Y, N856K, Q954H, N969K, L981F
    # S	gaps	H69-, V70-, G142-, V143-, Y144-, N211-
    fullseq = "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQDVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEHVNNSYECDIPIGAGICASYQTQTNSPRRARSVASQSIIAYTMSLGAENSVAYSNNSIAIPTNFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQEVFAQVKQIYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDCLGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNQNAQALNTLVKQLSSNFGAISSVLNDILSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPAICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVSGNCDVVIGIVNNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDDSEPVLKGVKLHYT"
    fullseq_list = list(fullseq)
    df = pd.read_csv('./mutations/omicron_ba1.csv', header=None)
    print(df)
    #content = r'hellow python, 123,end' 
    content_list = df.iloc[0, 1:].tolist()
    print(content_list)
    content_list02 = df.iloc[1, 1:7].tolist()
    print(content_list02)
    for content in content_list:
        print("="*10)
        print("content: {}".format(content))
        pattern = r'^\s*([A-Z])(\d+)([A-Z])$'
        result = re.match(pattern, content)
        if result: #none以外の場合
            mutation_position = int(result.group(2))
            print("metation_position: {}".format(mutation_position))
            new_aa = result.group(3)
            print("new_aa: {}".format(new_aa))
            print("fullseq_list[mutation_position-1] : {}".format(fullseq_list[mutation_position-1] ))
            fullseq_list[mutation_position-1] = new_aa
    for content in content_list02:
        print("="*10)
        print("content: {}".format(content))
        pattern = r'^\s*([A-Z])(\d+)(.+)$'
        result = re.match(pattern, content) 
        gap_count = 0
        if result: #none以外の場合
            gap_position = int(result.group(2))
            print("gap_position: {}".format(gap_position))
            del fullseq_list[gap_position-gap_count-1]
            gap_count += 1
    fullseq_mutation = "".join(fullseq_list)
    return fullseq_mutation

def test_data_gen_omicron_ba2():
    # S	reversionsToRoot	A67A, H69H, V70V, T95T, V143V, Y144Y, Y145Y, N211N, L212L, G446G
    # S	gaps	L24-, P25-, P26-
    fullseq = "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQDVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEHVNNSYECDIPIGAGICASYQTQTNSPRRARSVASQSIIAYTMSLGAENSVAYSNNSIAIPTNFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQEVFAQVKQIYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDCLGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNQNAQALNTLVKQLSSNFGAISSVLNDILSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPAICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVSGNCDVVIGIVNNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDDSEPVLKGVKLHYT"
    fullseq_list = list(fullseq)
    df = pd.read_csv('./mutations/omicron_ba2.csv', header=None)
    print(df)
    #content = r'hellow python, 123,end' 
    #content_list = df.iloc[0, 1:].tolist()
    #print(content_list)
    content_list02 = df.iloc[1, 1:4].tolist()
    print(content_list02)
    #for content in content_list:
    #    print("="*10)
    #    print("content: {}".format(content))
    #    pattern = r'^\s*([A-Z])(\d+)([A-Z])$'
    #    result = re.match(pattern, content)
    #    if result: #none以外の場合
    #        mutation_position = int(result.group(2))
    #        print("metation_position: {}".format(mutation_position))
    #       new_aa = result.group(3)
    #       print("new_aa: {}".format(new_aa))
    #       print("fullseq_list[mutation_position-1] : {}".format(fullseq_list[mutation_position-1] ))
    #        fullseq_list[mutation_position-1] = new_aa
    for content in content_list02:
        print("="*10)
        print("content: {}".format(content))
        pattern = r'^\s*([A-Z])(\d+)(.+)$'
        result = re.match(pattern, content) 
        gap_count = 0
        if result: #none以外の場合
            gap_position = int(result.group(2))
            print("gap_position: {}".format(gap_position))
            del fullseq_list[gap_position-gap_count-1]
            gap_count += 1
    fullseq_mutation = "".join(fullseq_list)
    return fullseq_mutation

def test_data_gen_omicron_ba3():
    # S	changes	A67V, T95I, Y145D, L212I, G339D, S371F, S373P, S375F, D405N, K417N, N440K, G446S, S477N, T478K, E484A, Q493R, Q498R, N501Y, Y505H, D614G, H655Y, N679K, P681H, N764K, D796Y, Q954H, N969K
    # S	gaps	H69-, V70-, G142-, V143-, Y144-, N211-
    fullseq = "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQDVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEHVNNSYECDIPIGAGICASYQTQTNSPRRARSVASQSIIAYTMSLGAENSVAYSNNSIAIPTNFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQEVFAQVKQIYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDCLGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNQNAQALNTLVKQLSSNFGAISSVLNDILSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPAICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVSGNCDVVIGIVNNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDDSEPVLKGVKLHYT"
    fullseq_list = list(fullseq)
    df = pd.read_csv('./mutations/omicron_ba3.csv', header=None)
    print(df)
    #content = r'hellow python, 123,end' 
    content_list = df.iloc[0, 1:].tolist()
    print(content_list)
    content_list02 = df.iloc[1, 1:7].tolist()
    print(content_list02)
    for content in content_list:
        print("="*10)
        print("content: {}".format(content))
        pattern = r'^\s*([A-Z])(\d+)([A-Z])$'
        result = re.match(pattern, content)
        if result: #none以外の場合
            mutation_position = int(result.group(2))
            print("metation_position: {}".format(mutation_position))
            new_aa = result.group(3)
            print("new_aa: {}".format(new_aa))
            print("fullseq_list[mutation_position-1] : {}".format(fullseq_list[mutation_position-1] ))
            fullseq_list[mutation_position-1] = new_aa
    for content in content_list02:
        print("="*10)
        print("content: {}".format(content))
        pattern = r'^\s*([A-Z])(\d+)(.+)$'
        result = re.match(pattern, content) 
        gap_count = 0
        if result: #none以外の場合
            gap_position = int(result.group(2))
            print("gap_position: {}".format(gap_position))
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
    print(df)
    #content = r'hellow python, 123,end' 
    content_list = df.iloc[0, 1:].tolist()
    print(content_list)
    content_list02 = df.iloc[2, 1:9].tolist()
    print(content_list02)
    for content in content_list:
        print("="*10)
        print("content: {}".format(content))
        pattern = r'^\s*([A-Z])(\d+)([A-Z])$'
        result = re.match(pattern, content)
        if result: #none以外の場合
            mutation_position = int(result.group(2))
            print("metation_position: {}".format(mutation_position))
            new_aa = result.group(3)
            print("new_aa: {}".format(new_aa))
            print("fullseq_list[mutation_position-1] : {}".format(fullseq_list[mutation_position-1] ))
            fullseq_list[mutation_position-1] = new_aa
    for content in content_list02:
        print("="*10)
        print("content: {}".format(content))
        pattern = r'^\s*([A-Z])(\d+)(.+)$'
        result = re.match(pattern, content) 
        gap_count = 0
        if result: #none以外の場合
            gap_position = int(result.group(2))
            print("gap_position: {}".format(gap_position))
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
    print(df)
    #content = r'hellow python, 123,end' 
    content_list = df.iloc[0, 1:].tolist()
    print(content_list)
    content_list02 = df.iloc[1, 1:5].tolist()
    print(content_list02)
    for content in content_list:
        print("="*10)
        print("content: {}".format(content))
        pattern = r'^\s*([A-Z])(\d+)([A-Z])$'
        result = re.match(pattern, content)
        if result: #none以外の場合
            mutation_position = int(result.group(2))
            print("metation_position: {}".format(mutation_position))
            new_aa = result.group(3)
            print("new_aa: {}".format(new_aa))
            print("fullseq_list[mutation_position-1] : {}".format(fullseq_list[mutation_position-1] ))
            fullseq_list[mutation_position-1] = new_aa
    for content in content_list02:
        print("="*10)
        print("content: {}".format(content))
        pattern = r'^\s*([A-Z])(\d+)(.+)$'
        result = re.match(pattern, content) 
        gap_count = 0
        if result: #none以外の場合
            gap_position = int(result.group(2))
            print("gap_position: {}".format(gap_position))
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
    print(df)
    #content = r'hellow python, 123,end' 
    content_list = df.iloc[0, 1:].tolist()
    print(content_list)
    content_list02 = df.iloc[2, 1:9].tolist()
    print(content_list02)
    for content in content_list:
        print("="*10)
        print("content: {}".format(content))
        pattern = r'^\s*([A-Z])(\d+)([A-Z])$'
        result = re.match(pattern, content)
        if result: #none以外の場合
            mutation_position = int(result.group(2))
            print("metation_position: {}".format(mutation_position))
            new_aa = result.group(3)
            print("new_aa: {}".format(new_aa))
            print("fullseq_list[mutation_position-1] : {}".format(fullseq_list[mutation_position-1] ))
            fullseq_list[mutation_position-1] = new_aa
    for content in content_list02:
        print("="*10)
        print("content: {}".format(content))
        pattern = r'^\s*([A-Z])(\d+)(.+)$'
        result = re.match(pattern, content) 
        gap_count = 0
        if result: #none以外の場合
            gap_position = int(result.group(2))
            print("gap_position: {}".format(gap_position))
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
    print(df)
    #content = r'hellow python, 123,end' 
    content_list = df.iloc[0, 1:].tolist()
    print(content_list)
    content_list02 = df.iloc[1, 1:9].tolist()
    print(content_list02)
    for content in content_list:
        print("="*10)
        print("content: {}".format(content))
        pattern = r'^\s*([A-Z])(\d+)([A-Z])$'
        result = re.match(pattern, content)
        if result: #none以外の場合
            mutation_position = int(result.group(2))
            print("metation_position: {}".format(mutation_position))
            new_aa = result.group(3)
            print("new_aa: {}".format(new_aa))
            print("fullseq_list[mutation_position-1] : {}".format(fullseq_list[mutation_position-1] ))
            fullseq_list[mutation_position-1] = new_aa
    for content in content_list02:
        print("="*10)
        print("content: {}".format(content))
        pattern = r'^\s*([A-Z])(\d+)(.+)$'
        result = re.match(pattern, content) 
        gap_count = 0
        if result: #none以外の場合
            gap_position = int(result.group(2))
            print("gap_position: {}".format(gap_position))
            del fullseq_list[gap_position-gap_count-1]
            gap_count += 1
    fullseq_mutation = "".join(fullseq_list)
    return fullseq_mutation

def data_loader(x_data_tr, y_data_tr, x_data_val, y_data_val):
    xtr = torch.tensor(x_data_tr, dtype=torch.float32)
    ytr = torch.tensor(y_data_tr, dtype=torch.float32)
    print("len of xtr: {}".format(len(xtr)))
    print("len of ytr: {}".format(len(ytr)))
    train_dataset = torch.utils.data.TensorDataset(xtr, ytr)
    print("train_dataset"+"#"*20)
    print(len(train_dataset))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                            batch_size=32,#batch_size=64, batch_size=512,  4 is best.1. 2.
                                            shuffle=False)
    xval = torch.tensor(x_data_val, dtype=torch.float32)
    yval = torch.tensor(y_data_val, dtype=torch.float32)
    print("len of xval: {}".format(len(xval)))
    print("len of yval: {}".format(len(yval)))
    #yte = torch.tensor(y_test, dtype=torch.int64)
    val_dataset = torch.utils.data.TensorDataset(xval, yval)
    #テスト用 Dataloder
    val_dataloader = torch.utils.data.DataLoader(val_dataset, 
                                            batch_size=64,#batch_size=64, batch_size=512. 1. 8.
                                            shuffle=False)
    #test_loader = test_dataloader
    print("data loading has completed!")
    return train_dataloader, val_dataloader 

def test_data_loader(x_data_test, y_data_test):
    xtest = torch.tensor(x_data_test, dtype=torch.float32)
    ytest = torch.tensor(y_data_test, dtype=torch.float32)
    test_dataset = torch.utils.data.TensorDataset(xtest, ytest)
    #テスト用 Dataloder
    test_dataloader = torch.utils.data.DataLoader(test_dataset, 
                                            batch_size=1,#batch_size=64, batch_size=512. 1. 8.
                                            shuffle=False)
    #test_loader = test_dataloader
    print("data loading has completed!")
    return test_dataloader

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout: float = 0.1, max_len: int = 5000) -> None:
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        #pe_p = self.pe[:, :x.size(1)]
        pe_p = self.pe[:, :x.size(0)]
        x = x + pe_p
        return self.dropout(x)

#モデルに入力するために次元を拡張する
class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        self.tokenConv = nn.Linear(c_in, d_model) 

    def forward(self, x):
        x = self.tokenConv(x)
        return x
    
class Transformer(nn.Module):
    def __init__(self, num_encoder_layers, num_decoder_layers,
        d_model, d_input, d_output,
        dim_feedforward, dropout, nhead): #dim_ff = 512. dropout =0.1.
        
        super(Transformer, self).__init__()
        

        #エンべディングの定義
        self.token_embedding_src = TokenEmbedding(d_input, d_model)
        self.token_embedding_tgt = TokenEmbedding(d_output, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)
        
        #エンコーダの定義
        encoder_layer = TransformerEncoderLayer(d_model=d_model, 
                                                nhead=nhead, 
                                                dim_feedforward=dim_feedforward,
                                                dropout=dropout,
                                                batch_first=True,
                                                activation='gelu'
                                                )
        encoder_norm = LayerNorm(d_model)
        self.transformer_encoder = TransformerEncoder(encoder_layer, 
                                                    num_layers=num_encoder_layers,
                                                    norm=encoder_norm
                                                    )
        
        #デコーダの定義
        decoder_layer = TransformerDecoderLayer(d_model=d_model, 
                                                nhead=nhead, 
                                                dim_feedforward=dim_feedforward,
                                                dropout=dropout,
                                                batch_first=True,
                                                activation='gelu'
                                                )
        decoder_norm = LayerNorm(d_model)
        self.transformer_decoder = TransformerDecoder(decoder_layer, 
                                                    num_layers=num_decoder_layers, 
                                                    norm=decoder_norm
                                                    )
        
        #出力層の定義
        #self.output = nn.Linear(d_model, d_output)
        
        self.fc = nn.Linear(d_model, d_output) #it was 18, 2.
        #self.fc1 = nn.Linear(d_model*d_model, 512)
        #self.fc2 = nn.Linear(512, 1)
        #self.sm = nn.Softmax(dim=1) #dim was 2. why??
        #self.sig = F.sigmoid(x)

    def forward(self, src, tgt, mask_src, mask_tgt):
        print("size of src in Transformer forwawrd.: {}".format(src.size()))
        token_embedding_src = self.token_embedding_src(src)
        embedding_src = self.positional_encoding(token_embedding_src)
        #print("embedding_src.")
        #print(embedding_src)
        #print(embedding_src.size())
        memory = self.transformer_encoder(embedding_src, mask_src)
        memory = np.squeeze(memory, 0)
        memory = memory.reshape(memory.shape[0], -1)
        #print("memory.")
        #print(memory)
        #print(memory.size())
        x = self.fc(memory)
        #x = self.fc1(memory)
        #x = self.fc2(x)
        #x = self.sm(x)
        x = F.sigmoid(x)
        return x


    def encode(self, src, mask_src):
        return self.transformer_encoder(self.positional_encoding(self.token_embedding_src(src)), mask_src)

    def decode(self, tgt, memory, mask_tgt):
        return self.transformer_decoder(self.positional_encoding(self.token_embedding_tgt(tgt)), memory, mask_tgt)
class NN(nn.Module):
    def __init__(self, d_model, d_input, d_output):
        self.d_model = d_model
        self.d_input = d_input
        self.d_output = d_output
        self.fc1 = nn.Linear(d_input, 512)
        self.fc2 = nn.Linear(512, 32)
        self.fc = nn.Linear(32, d_output)
    def forward(self, input_data):    
        x = self.fc1(input_data)
        x = self.fc2(x)
        x = self.fc(x)
        x = F.sigmoid(x)
        return x


def create_mask(src, tgt, device):
    seq_len_src = src.shape[0]
    seq_len_tgt = tgt.shape[0]

    mask_tgt = generate_square_subsequent_mask(seq_len_tgt).to(device)
    mask_src = generate_square_subsequent_mask(seq_len_src).to(device)

    return mask_src, mask_tgt


def generate_square_subsequent_mask(seq_len):
    mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
    return mask


def train(model, data_provider, optimizer, criterion, device, epoch):
    model.train()
    total_loss = []
    
    tgt_list_train = []
    output_list_train = []

    #correct_list_train = []
    h = 0
    for src, tgt in data_provider:

        print("epoch: {}, {}th train phase".format(epoch, h)+"="*20)
        
        src = src.float().to(device)
        tgt = tgt.float().to(device)

        #print("tgt")
        #print(tgt)

        input_tgt = None
        mask_src = None
        mask_tgt = None 
        #print("train src")
        #print(src)
        output = model(
            src=src, tgt=input_tgt, 
            mask_src=mask_src, mask_tgt=mask_tgt
        )
        #print("output in train.")
        #print(output)
        
        tgt_list_train.append(tgt)
        output_list_train.append(output)
        
        optimizer.zero_grad()


        loss = criterion(output, tgt)
        #print("loss in train.")
        #print(loss)
        #loss = criterion(output, tgt)
        #loss.retain_grad()
        
        loss.backward()

        total_loss.append(loss.cpu().detach())
        #print("total_loss in train.")
        #print(total_loss)

        optimizer.step()

        h = h + 1
        

        
    return np.average(total_loss), tgt_list_train, output_list_train

"""
def evaluate(flag, model, data_provider, criterion, device, epoch, l, protease):
    model.eval()
   total_loss = []

    tgt_list_valid = []
    output_list_valid = []

    #correct_list_valid = []

    with torch.no_grad():
        h = 0
        for src, tgt in data_provider:
            print("epochs: {}, {}th valid phase".format(epoch, h)+"="*20)
            
            src = src.float().to(device)
            tgt = tgt.float().to(device)

            seq_len_src = src.shape[1]

            #print("evaluate src")
            #print(src)
            #print(src.shape)

            tgt02 = torch.unsqueeze(tgt[-1], dim=0)

            mask_src = None
            mask_tgt = None
            output = model(src, tgt, mask_src, mask_tgt)
            #print("output in valid.")
            #print(output)
            output02 = output.tolist()
            #print("output02 in valid.")
            #print(output02)
            #print(len(output02))
            tgt_list_valid.append(tgt)
            output_list_valid.append(output)


            loss = criterion(output, tgt)
            total_loss.append(loss.cpu().detach())
            h = h + 1
        
    if flag=='test':
        true = torch.cat((src, tgt), dim=1)
        pred = torch.cat((src, output), dim=1)
        plt.plot(true.squeeze().cpu().detach().numpy(), label='true')
        plt.plot(pred.squeeze().cpu().detach().numpy(), label='pred')
        plt.legend()
        #plt.savefig('test.pdf')
        #"C:/Users/ysasaki.ADC/Documents/workstation_ysasaki/python/LSTM/VR-ML/MLprogram/labeled_data/flow2-2nd-move-data/outputdir"
        plt.savefig("./outputdir_rf_reg_priority/{}_{}/test.pdf".format(l, protease))
    #return np.average(total_loss), tgt_list_valid, output_list_valid, correct_list_valid
    return np.average(total_loss), tgt_list_valid, output_list_valid
"""

def predict(val_dataloader, model, device, model_path):
    test_dataloader = val_dataloader
    #model_path = "c:/Users/ysasaki.ADC/Documents/workstation_ysasaki/theme/NEDO-XRAI/MR3-ware/share_dataset_for_ysasaki/share_dataset_for_ysasaki/input_data7/outputdir/tf_checkpoint/3_2025-01-16_397756_transformer_checkpoint.pt"
    #model_path = "./outputdir/tf_checkpoint/"
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device))) 
    model.eval()
    with torch.no_grad():
        # output_all = torch.zeros(1, 4)
        output_all = np.zeros((1, 1))
        for i in range(len(test_dataloader)):
        #for j in range(1):
            #for i in range(len(test_dataloader)):
            #if j == 1:
            #    break
            if i == 0:
                batch_iterator = iter(test_dataloader)  # イテレータに変換, dataloaders_dict["test"]
            #print("="*20+"batch_iterator"+"="*20)
            #print(batch_iterator)
            tdata, labels = next(batch_iterator)  # 1番目の要素を取り出す
            #print("="*20+"test data {}".format(i)+"="*20)
            #print(tdata)
            #print(tdata.shape)
            print("="*20+"test labels {}: ".format(i)+"="*20)
            #print(labels)
            #print(len(labels))
            # GPUを使用する場合は明示的に指定
            src = tdata.to(device)
            tgt = labels.to(device)
            #tgt02 = torch.unsqueeze(tgt[-1], dim=0)
            #print("tgt02")
            #print(tgt02)
            #print("src[-1:]")
            #print(src[-1:])
            #input_tgt = torch.cat((src[-1:],tgt02), dim=1) # dim was 1.
            #print("input_tgt")
            #print(input_tgt)
            # 出力
            #mask_src, mask_tgt = create_mask(src, input_tgt, device)
            mask_src = None
            mask_tgt = None
            input_tgt = None 
            outputs = model(
                src=src, tgt=tgt, 
                mask_src=mask_src, mask_tgt=mask_tgt
            ) #順伝播
            outputs = np.array(outputs.to('cpu'))
            print(outputs)
            #outputs = torch.squeeze(outputs, dim=0)
            #output_all = torch.cat([output_all, outputs], axis=0)
            output_all = np.concatenate([output_all, outputs])
            #print("output_all in progress.")
            #print(output_all)
            #print(output_all.shape)
    output_all = output_all[1:]
    print("output_all.")
    print(output_all)
    print(output_all.shape)
    return output_all

if __name__ == '__main__':
    start = time.time()
    
    main()

    end = time.time()
    time_diff = end - start
    print(time_diff)
    print(time_diff/60)
    print(time_diff/60/60)
    print(time_diff/60/60/24)
    
    dt_now = datetime.datetime.now()
    print(dt_now)

