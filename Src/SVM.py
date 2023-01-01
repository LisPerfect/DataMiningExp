from sklearn.preprocessing import KBinsDiscretizer
from sklearn.svm import SVC
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
df = read_csv('./z-score.csv')
df.dropna(inplace=True)
Accu = []
y = df['quality']
df = df[['volatile acidity', "chlorides", 'total sulfur dioxide', 'density', 'alcohol']]
cols = ['volatile acidity', "chlorides", 'total sulfur dioxide', 'density', 'alcohol']
def train_and_test(df):
    discretizer = KBinsDiscretizer(n_bins=9, encode='ordinal')
    discretizer.fit(np.array(df).reshape(-1, 1))
    x_trans = discretizer.transform(np.array(df).reshape(-1, 1))
    
    discretizer = KBinsDiscretizer(n_bins=4, encode='ordinal')
    discretizer.fit(np.array(y).reshape(-1, 1))
    y_trans = discretizer.transform(np.array(y).reshape(-1, 1))
    
    X_train, X_test, y_train, y_test = train_test_split(x_trans, y_trans, test_size=0.2)
    model = SVC(kernel='rbf', tol=1)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    Accu.append(accuracy)
for i in cols:
    df_train = df[i]
    train_and_test(df_train)
print(Accu)