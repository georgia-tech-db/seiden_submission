from collections import defaultdict

import pandas as pd
import numpy as np
import scipy.special
import feather

from supg import datasource


def load_jackson_source(probs_fname, csv_fname, obj_name):
    # y_true
    df_csv = pd.read_csv(csv_fname)
    df_csv = df_csv[df_csv['object_name'] == obj_name]
    groups = defaultdict(list)
    for row in df_csv.itertuples():
        groups[row.frame].append(row)

    Y = [len(groups[i]) for i in range(max(groups))]
    Y = np.array(Y, dtype=np.int64)
    y_true = np.minimum(Y, 1)

    # y_probs
    preds = np.fromfile(probs_fname, dtype=np.float32).reshape(-1, 2)
    preds = scipy.special.softmax(preds, axis=1)
    preds = preds[:, 1]

    y_prob = preds[0:len(y_true)]
    y_true = y_true[0:len(y_prob)]

    data = {'id': list(range(len(y_prob))),
            'proxy_score': y_prob,
            'label': y_true}
    df = pd.DataFrame(data)
    return df

def get_jackson_source(drop_p=None, seed=None) -> datasource.DataSource:
    # probs_fname = '../../data/jackson/2017-12-17-s50-bin.out'
    # csv_fname = '../../data/jackson/jackson-town-square-2017-12-17.csv'
    # obj_name = 'car'
    # df = datasource.load_jfile_df(probs_fname, csv_fname, obj_name)
    # source = datasource.DFDataSource(df, drop_p=drop_p, seed=seed)
    df = pd.read_feather('../../data/jackson/2017-12-17.feather')
    source = datasource.DFDataSource(df, drop_p=drop_p, seed=seed)
    return source

def load_csv_source(csv_fname) -> datasource.DataSource:
    df = pd.read_csv(csv_fname)
    df['label'] = df['label'].astype('float32')
    source = datasource.DFDataSource(df)
    return source

def get_imagenet_source() -> datasource.DataSource:
    return load_csv_source('../../data/imagenet/source.csv')

def get_onto_source() -> datasource.DataSource:
    return load_csv_source('../../data/onto/source.csv')

def get_tacred_source() -> datasource.DataSource:
    return load_csv_source('../../data/tacred/source.csv')
