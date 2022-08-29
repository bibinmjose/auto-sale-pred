import sys
import embedding as emb
import pandas as pd
import numpy as np
import os
import argparse
import yaml
import pickle
import seaborn as sns

import matplotlib.pyplot as plt
plt.style.use('ggplot')


def load_file(config):
    train = pd.read_csv(
        config['train'], 
        )

    train.info()
    return train

def make_plots(dfs):
    x,y = 0,1
    
    for prefix, df in dfs.items():

        fig, ax = plt.subplots()

        ax = sns.scatterplot(
            x=f'{prefix}_embedding_{x}',
            y=f'{prefix}_embedding_{y}', 
            data=df, legend='auto')

        emb.label_point(
            f'{prefix}_embedding_{x}', 
            f'{prefix}_embedding_{y}', df, ax)

        fig.savefig(f'./nbs/{prefix}_enc.png')

    

def make_embeddings(config):
    train = load_file(config)
    y_train = train.Sale_Price
    
    cat = config['cat']
    #create embeddings
    embedding_info = emb.get_embedding_info(train, 
                                            categorical_variables=cat, max_n=2)
    print(f'Embedding Info: {embedding_info}')
    
    X_encoded, encoders = emb.get_label_encoded_data(train.loc[:,cat])
    embeddings = emb.get_embeddings(X_encoded, y_train, categorical_embedding_info=embedding_info, 
                               is_classification=False, epochs=100,batch_size=256)
    emb_dfs = emb.get_embeddings_in_dataframe(embeddings=embeddings, encoders=encoders)
    make_plots(emb_dfs)
    
    ## save embeddings
    with open(os.path.join(config['dir'],'embeddings.pkl'), 'wb') as handle:
        pickle.dump(emb_dfs, handle)

    with open(os.path.join(config['dir'],'encoders.pkl'), 'wb') as handle:
        pickle.dump(encoders, handle)
    
if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Script to Process Data',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-c','--config', type=str,
                        default= './config.yml',
                        help='Config path')

    parser.add_argument('-t', '--train', type=str,
                        default='./data/train_2021-12-24.csv',
                        help='data directory')
    parser.add_argument('-d', '--dir', type=str,
                        default='./data',
                        help='data directory')
    
    args = parser.parse_args()
    
    config = yaml.safe_load(open(args.config))
    for k,val in vars(args).items():
        config[k]=val

    print(config)
    make_embeddings(config)