from sklearn.svm import SVR
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
df = read_csv('./z-score.csv')
df.dropna(inplace=True)
y = df['quality']
result = dict()
df = df[['volatile acidity', "chlorides", 'total sulfur dioxide', 'density', 'alcohol']]
cols = ['volatile acidity', "chlorides", 'total sulfur dioxide', 'density', 'alcohol']
def train_and_test(df):
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2)
    reg = SVR(kernel='rbf')
    X_train = np.array(X_train).reshape(-1, 1)
    #y_train = np.array(y_train).reshape(-1, 1)
    X_test = np.array(X_test).reshape(-1, 1)
    y_test = np.array(y_test).reshape(-1, 1)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, mae, r2
for i in cols:
    df_train = df[i]
    mse, mae, r2 = train_and_test(df_train)
    l = [mse, mae, r2]
    result[i] = l
r = pd.DataFrame.from_dict(result)
print(r)