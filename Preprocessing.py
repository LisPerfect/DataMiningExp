import pandas as pd
import numpy as np
from pandas import read_excel
data = read_excel('./white_wine_quality.xls')
#print(data)
data.drop_duplicates(inplace=True)
#print(data)
data.dropna(inplace=True)
print(data)

# 通过Z-Score方法判断异常值
df_zscore = data.copy()  # 复制一个用来存储Z-score得分的数据框
cols = data.columns  #  获得列表框的列名
for col in cols:
    df_col = data[col]  #  得到每一列的值
    z_score = (df_col - df_col.mean()) / df_col.std()  #  计算每一列的Z-score得分
    df_zscore[col] = z_score.abs() > 3  # 判断Z-score得分是否大于3，如果是则是True，否则为False
df_drop_outlier = data.copy()
for col in cols:    
    df_drop_outlier = data[df_zscore[col] == False]
print(df_drop_outlier)
df_drop_outlier.to_csv('after_preprocessing.csv')
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
for col in cols:
    df_col = df_drop_outlier[col]
    z_score = (df_col - df_col.mean()) / df_col.std()
    df_zscore[col] = z_score
df_zscore.to_csv('./z-score.csv')