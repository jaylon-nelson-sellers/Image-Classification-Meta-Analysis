from DataLogger import DataLogger
from pandas import DataFrame
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
import xgboost as xgb
import torch
import pandas as pd
import numpy as np

from EasyTorch.EasyConvNet import EasyConvNet
from EasyTorch.EasyNeuralNet import EasyNeuralNet
from EvaluateTests import evaluate_image_test

def dummy_test(dataset_name:str,dataset_data:DataFrame):
    dl = DataLogger(dataset_name,0,0,0,0)
    dummy_tests = [
            DummyClassifier(),
            DummyClassifier(strategy='uniform')
        ]
    for d in dummy_tests:
        evaluate_image_test(d,dataset_data,dl)

def sk_test(dataset_name:str,dataset_data:DataFrame):
    dl = DataLogger(dataset_name,0,0,0,0)
    models = [
        # Linear Models
        LogisticRegression(random_state=42),
        # Decision Trees
        DecisionTreeClassifier(random_state=42),

        # Nearest Neighbors
        KNeighborsClassifier(n_neighbors=5),

        # Support Vector Machines
        #SVC(kernel='rbf', random_state=42,),
        LinearSVC(random_state=42),
        
        
        # Discriminant Analysis
        LinearDiscriminantAnalysis(),
        QuadraticDiscriminantAnalysis()]
    for m in models:
        evaluate_image_test(m,dataset_data,dl)

def xgb_tests(dataset_name:str,dataset_data:DataFrame):
    dl = DataLogger(dataset_name,0,0,0,0)
    estimators = [2,4,8,16,32,64,128,256,512]
    for e in estimators:
        #temp =xgb.XGBRFClassifier(n_estimators=e,learning_rate=0.1,device='cuda',n_jobs=-1)
        #evaluate_image_test(temp,dataset_data,dl)
        temp = xgb.XGBClassifier(
                n_estimators=e,
                learning_rate=0.1,
                max_depth=5,
                device='cuda', 
                n_jobs=-1,

            )
        evaluate_image_test(temp,dataset_data,dl,tensor_check=True)
        

def nn_tests(dataset_name:str, dataset_data:DataFrame):
    dl = DataLogger(dataset_name,0,0,0,0)
    if dataset_name == "mnist" or dataset_name == "fashion" or dataset_name == "cifar10":
        output_size = 10
    if dataset_name == "coarse":
        output_size = 20
    if dataset_name == "fine":
        output_size = 100
    hidden_sizes = [2,4,8,16,32,64,128]
    dropout = 0
    for h in hidden_sizes:
        layers = (h,h)
        model = EasyNeuralNet(output_size,layers,dropout,criterion_str="CrossEntropyLoss",problem_type=0,batch_norm=True,learning_rate=.001,image_bool=False,verbose=True)
        evaluate_image_test(model,dataset_data,dl,tensor_check=True)

def cnn_tests(dataset_name:str, dataset_data:DataFrame):
    dl = DataLogger(dataset_name,0,0,0,0)
    dims = 2
    if dataset_name == "mnist" or dataset_name == "fashion" or dataset_name == "cifar10":
        output_size = 10
        input_size = 784
    if dataset_name == "coarse":
        output_size = 20
        input_size = 784
    if dataset_name == "fine":
        output_size = 100
        input_size = 784
    channels = [16,32]
    conv_blocks = [1,2,3]
    pool_blocks = [1,2,3]
    dropout = 0
    dense = 2
    for pool in pool_blocks:
        for conv in conv_blocks:
            for c in channels:
                model = EasyConvNet(input_size,output_size,dims,c,conv,pool,dense,dropout,
                                    criterion_str="CrossEntropyLoss",problem_type=0,batch_norm=True,learning_rate=.001,verbose=True)
                evaluate_image_test(model,dataset_data,dl,tensor_check=True)