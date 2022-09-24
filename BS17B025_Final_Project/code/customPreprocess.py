import numpy as np
import pandas as pd
from pandas import get_dummies

target = 'Slowness in traffic (%)'

def create_day(df):
    '''
    A method that processes the Hour feature of the dataframe
    to create a new feature 'Day'.
    '''
    days = ['MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY']
    idx_ub = list(np.arange(26, 135, 27))
    idx_lb = [0] + [idx + 1 for idx in idx_ub[0:-1]]

    #Encoding the index bounds of dataframe for corresponding days.
    days_index = {k: v for (k, v) in zip(days, zip(idx_lb, idx_ub))}

    #Creating the day column.
    if 'Day' not in df.columns:
        df['Day'] = 0

    #Encoding is done such that Monday->1, Tuesday->2,...
    for day, idx_bnd in days_index.items():
        lb, ub = idx_bnd
        df.loc[lb:ub, 'Day'] = days.index(day) + 1

    return df

def categorise_hour(df):
    '''
    A method that creates categories on Hour feature.
    '''
    times = ['Morning', 'After-noon', 'Evening', 'Night']
    hour_code = {
        'Morning': (1, 7),
        'After-noon': (8, 14),
        'Evening': (15, 21),
        'Night': (21, 27)
    }
    #Creating a feature that indicates the time of day.
    df['Time of day'] = 0

    #Morning ->1, After-noon->2, Evening->3, Night->4.
    for time, (h1, h2) in hour_code.items():
        filt = (df['Hour (Coded)'] >= h1) & (df['Hour (Coded)'] <= h2)
        df.loc[filt, 'Time of day'] = times.index(time) + 1
        
    #Dropping the hour column.
    df = df.drop(columns=['Hour (Coded)'], axis=1)
    
    return df


def one_hot_encode(df, category_features=['Day', 'Time of day']):
    '''
    A method that does one hot encoding on category features.
    '''
    dummies = {}
    #One hot encoding using pandas get_dummies.
    for feature in category_features:
        dummies[feature] = get_dummies(df[feature],
                                       prefix=feature,
                                       prefix_sep=' ')
        
    #Combining dummies with the dataframe.
    df = df.join(list(dummies.values()))
    
    #Dropping the initial columns that were one-hot encoded.
    df = df.drop(columns=category_features, axis=1)

    return df

def preprocess(dataframe, apply_log=False):
    '''
    A method that does all of the above preprocessing.
    '''
    #Making a copy of the dataframe.
    df = dataframe.copy()

    #Conversion of target variable to float.
    df[target] = df[target].apply(lambda x: str(x).replace(',', '.')).astype('float')

    #Creating the day feature.
    df = create_day(df)

    #Categorising the hour feature.
    df = categorise_hour(df)

    #One-hot encoding of category variables.
    df = one_hot_encode(df)

    #Splitting the features and target variable.
    y = df[target]

    if apply_log:
        y = np.log(y)
    
    features = df.columns.drop(target)
    X = df[features]

    return X,y