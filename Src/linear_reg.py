from sklearn.linear_model import LinearRegression
from pandas import read_csv
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import KBinsDiscretizer

df = read_csv('./z-score.csv')
df.dropna(inplace=True)
mse = dict()
y = df['quality']
df = df[['volatile acidity', "chlorides", 'total sulfur dioxide', 'density', 'alcohol']]
cols = ['volatile acidity', "chlorides", 'total sulfur dioxide', 'density', 'alcohol']

def train_and_test(df, col):
    discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal')
    discretizer.fit(np.array(df).reshape(-1, 1))
    x_trans = discretizer.transform(np.array(df).reshape(-1, 1))
    
    discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal')
    discretizer.fit(np.array(y).reshape(-1, 1))
    y_trans = discretizer.transform(np.array(y).reshape(-1, 1))
    X_train, X_test, y_train, y_test = train_test_split(x_trans, y_trans, test_size=0.2)
    X_train = np.array(X_train).reshape(-1, 1)
    X_test = np.array(X_test).reshape(-1, 1)
    y_train = np.array(y_train).reshape(-1, 1)
    y_train = y_train.ravel()
    y_test = np.array(y_test).reshape(-1, 1)
    y_test = y_test.ravel()
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    mseVal = float(mean_squared_error(y_test, y_pred))
    mse[col] = mseVal

for i in cols:
    df_train = df[i]
    train_and_test(df_train, i)
print(mse) 