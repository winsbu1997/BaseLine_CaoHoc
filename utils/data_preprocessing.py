import pandas as pd
import numpy as np
from sklearn import preprocessing
import random
import joblib

def load_data(file_path):
    return pd.read_csv(file_path, encoding="cp1252", engine='python')

def preprocess_data(df, save_data):
    df.columns = df.columns.str.strip()
    df['Label'].unique()
    print(df['Label'].unique())
    df['Label'].value_counts()
    print(df['Label'].value_counts())
    
    df.replace('Infinity', -1, inplace=True)
    # df[["Flow Bytes/s", "Flow Packets/s"]] = df[["Flow Bytes/s", "Flow Packets/s"]].apply(pd.to_numeric)
    df.replace([np.inf, -np.inf, np.nan], -1, inplace=True)
    string_features = list(df.select_dtypes(include=['object']).columns)
    string_features.remove('Label')
    string_features
    le = preprocessing.LabelEncoder()
    df[string_features] = df[string_features].apply(lambda col: le.fit_transform(col))
    
    benign_total = 0
    attack_total = 0
    if("2017" in save_data):
        benign_total = len(df[df['Label'] == "BENIGN"])
        attack_total = len(df[df['Label'] != "BENIGN"])
    else:
        benign_total = len(df[df['Label'] == "Benign"])
        attack_total = len(df[df['Label'] != "Benign"])
    print("ToTal begnign: " + str(benign_total))
    print("ToTal attack: " + str(attack_total))
    #excluded = ['Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Protocol', 'Timestamp']
    #df = df.drop(columns=excluded, errors='ignore')
    #excluded2 = ['Init_Win_bytes_backward', 'Init_Win_bytes_forward']
    #df = df.drop(columns=excluded2, errors='ignore')
    df.to_csv(save_data + "web_attacks.csv", index=False)