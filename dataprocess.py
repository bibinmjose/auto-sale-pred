import pandas as pd
import numpy as np
import os
import argparse
import yaml


def load_file(config):
    raw_sale = pd.read_csv(
        config['input_file'], 
        usecols=config['cols']
        )

    print(f"{30*'*'}\nSummary Stats\n{raw_sale.describe().round().T}")

    _fltr = raw_sale.Advertised_Date.str[:4].astype(int)>=2014
    sale_adv_fltr= raw_sale.loc[_fltr,:].copy()

    sale_adv_fltr['Sold_Date'] = pd.to_datetime(sale_adv_fltr['Sold_Date'])
    sale_adv_fltr['Advertised_Date'] = pd.to_datetime(sale_adv_fltr['Advertised_Date'])

    sale_date_fltr = sale_adv_fltr.query('Advertised_Date <= Sold_Date')
    print(f"{30*'*'}\nFiltered-in Data %: {sale_date_fltr.shape[0]/raw_sale.shape[0]:2.5f}")
    print(f"{30*'*'}\nSummary Stats After Process\n{sale_date_fltr.describe().round().T}")
    return sale_date_fltr

def feat_engg(df):
    df = df.assign(
    make_date = pd.to_datetime(df.Year_Group, format='%Y'), # make_date is approx to year start for age calc
    )
    df = df.assign(
        inventory_days=(df.Sold_Date-df.Advertised_Date).dt.days,
        age_list_day=(df.Advertised_Date-df.make_date).dt.days,
        age_sell_day=(df.Sold_Date-df.make_date).dt.days,
        sld_wknum = df.Sold_Date.dt.isocalendar().week,
        adv_wknum = df.Advertised_Date.dt.isocalendar().week,
    )
    df = df.assign(
        km_per_day = (df.Odometer/df.age_sell_day),
        fuel_per_power = (df.Fuel_Urban/df.Power),
        odo_per_fuel = (df.Odometer/df.Fuel_Urban),
        odo_per_power = (df.Odometer/df.Power),
        pow_times_odo = (df.Odometer*df.Power),
    )
    df = df.query('inventory_days>=0')
    print(f"{30*'*'}\nSummary Stats after FEATURE ENGINEERING:\n{df.describe().round().T}")
    return df

def test_train_split(all_df):
    cols= [c for c in all_df.columns if c not in ['Advertised_Date', 'Sold_Date','make_date']]
    # save entire dataset
    all_df.to_csv(os.path.join(config['dir'],'sales.csv'), index=False)
    splt_idx = int(all_df.shape[0] * config['train_prop'])

    # split train-test with Sold_Date to mimic model in prod
    train = all_df.sort_values('Sold_Date').iloc[:splt_idx,:]
    test = all_df.sort_values('Sold_Date').iloc[splt_idx:,:]
    max_dt_train, max_dt_test = train.Sold_Date.max().strftime('%Y-%m-%d'), test.Sold_Date.max().strftime('%Y-%m-%d')
    print(f"{30*'*'}\nTrain Max Date: {max_dt_train, train.shape}\nTest Max Date: {max_dt_test, test.shape}")

    train.loc[:,cols].to_csv(os.path.join(config['dir'],f'train_{max_dt_train}.csv'), index=False)
    test.loc[:,cols].to_csv(os.path.join(config['dir'],f'test_{max_dt_test}.csv'), index=False)
    return  train, test


def process(config):
    data_frame = load_file(config)
    feat = feat_engg(data_frame)
    train, test = test_train_split(feat)

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Script to Process Data',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-c','--config', type=str,
                        default= './config.yml',
                        help='Config path')

    parser.add_argument('-d', '--dir', type=str,
                        default='./data',
                        help='data directory')
    
    args = parser.parse_args()

    # if not os.path.exists(args.dir):
    #     raise NotADirectoryError
    
    config = yaml.safe_load(open(args.config))
    for k,val in vars(args).items():
        config[k]=val

    print(config)
    process(config)