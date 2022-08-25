# import libraries

import argparse
import pandas as pd
import numpy as np
import pickle
import os
import pdb
import yaml
import xgb
import optuna
from joblib import dump
from sklearn.impute import KNNImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score




def load_files(args):
    '''Load all datasets into a datadict
    '''
    _pth_trn = os.path.join(args.pre, args.train)
    _pth_emb = os.path.join(args.pre, 'embeddings.pkl')

    _dict=dict()
    _dict['train'] = pd.read_csv(_pth_trn)
    with open(_pth_emb, 'rb') as handle:
        _dict['embed'] = pickle.load(handle)

    return _dict

def get_embedded_feat(feat, encod_dict, feat_name='Family_Code'):
    '''Return embeddings for each feature column using encoded dict
    '''
    feat_df = feat.to_frame()
    encodings = encod_dict[feat_name].reset_index()
    feat_tranf = feat_df.merge(encodings, how='left', left_on=[feat_name], right_on=['index'], indicator=True)
    # print(feat_tranf._merge.value_counts())
    return feat_tranf.loc[:,[c for c in encodings.columns if c.startswith(feat_name)]]


def assemble_features(df, feat_cols, embeddings):
    '''Assemble all retrived embeddings and features into one dataframe
    '''
    _dflist = []
    
    for f in embeddings:
        _df = get_embedded_feat(df.loc[:,f], embeddings, feat_name=f)
        _dflist.append(_df)
        
    return pd.concat([df.loc[:,feat_cols]]+ _dflist, axis=1)

def impute_missing(df, out_dir):
    '''Returns imputed tesonr for train
    Saves the imputer to out_path.
    '''
    knn = KNNImputer()
    knn.fit(df)
    df_imputed = knn.transform(df)
    dump(knn, os.path.join(out_dir,'knnimputer.joblib'))
    return df_imputed


def objective(trial, X_train, y_train):
    '''Objective function for the optimizer
    '''
    param = {
        "silent": 1,
        "objective": "reg:squarederror",
        "eval_metric": r2_score,
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        "lambda": trial.suggest_loguniform("lambda", 1e-8, 1.0e6),
        "alpha": trial.suggest_loguniform("alpha", 1e-8, 1.0e6),
    }
    if param["booster"] == "gbtree" or param["booster"] == "dart":
        param["max_depth"] = trial.suggest_int("max_depth", 2, 5)
        param["eta"] = trial.suggest_loguniform("eta", 1e-8, 0.05)
        param["gamma"] = trial.suggest_loguniform("gamma", 1e-8, 1.0)
        param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        param["rate_drop"] = trial.suggest_loguniform("rate_drop", 1e-8, 1.0)
        param["skip_drop"] = trial.suggest_loguniform("skip_drop", 1e-8, 1.0)
    
    model = xgb.XGBRegressor(**param)
    
    return cross_val_score(model, X_train, y_train, n_jobs=-1, cv=10, scoring='r2').mean()

# train function
def train(args):
    '''Train runner to create a model.
    '''
    # load config and all data files
    config = yaml.safe_load(open(args.config))
    dataset = load_files(args)
    # assemble features: replace categorical with encoding
    X_train_df = assemble_features(
        dataset['train'], 
        config['feat_cols'], 
        dataset['embed']
        )
    y_train = dataset['train'].loc[:,'Sale_Price']
    # run imputation for missing values and save it
    X_train = impute_missing(X_train_df, args.out)

    # Create HPO and find best params
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)
    model = xgb.XGBRegressor(**study.best_params, feature_names=X_train_df.columns.tolist())
    model.fit(X_train, y_train)
    # save best model
    model.save_model(os.path.join(args.out,"xgb_model.json"))

    pdb.set_trace()

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Script for Train a model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-c','--config', type=str,
                        default= './config.yml',
                        help='Config path')

    parser.add_argument('-p', '--pre', type=str,
                        default='./data',
                        help='Prefix for all data files')
    
    parser.add_argument('-t', '--train', type=str,
                        default='train_2021-12-24.csv',
                        help='directory from which data')

    parser.add_argument('-o', '--out', type=str,
                        default='./out',
                        help='output directory')
    
    args = parser.parse_args()

    if not os.path.exists(args.out):
        os.makedirs(args.out)

    train(args)