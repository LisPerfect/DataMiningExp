import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import math
plt.rcParams["font.sans-serif"]=["SimHei"] #设置中文字体，防止乱码
plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题

from pandas import read_csv
df = read_csv('./after_preprocessing.csv')
df.drop(df.columns[0], inplace=True, axis=1)
df_z = read_csv('./z-score.csv')
df_z.drop(df_z.columns[0], inplace=True, axis=1)
cols = df.columns

fa, va, ca, rs, ch, fsc, fsd, de, ph, su, al, qu = df['fixed acidity'], df['volatile acidity'], df['citric acid'], df['residual sugar'], df['chlorides'], df['free sulfur dioxide'], df['total sulfur dioxide'], df['density'], df['pH'], df['sulphates'], df['alcohol'], df['quality']
fa_z, va_z, ca_z, rs_z, ch_z, fsc_z, fsd_z, de_z, ph_z, su_z, al_z, qu_z = df_z['fixed acidity'], df_z['volatile acidity'], df_z['citric acid'], df_z['residual sugar'], df_z['chlorides'], df_z['free sulfur dioxide'], df_z['total sulfur dioxide'], df_z['density'], df_z['pH'], df_z['sulphates'], df_z['alcohol'], df_z['quality']

plt.figure(figsize=(10,6))#c
#Volin plot for each data
plt.subplot(341)
plt.violinplot(fa, showmeans=True)
plt.title('Fixed acidity')
plt.subplot(342)
plt.violinplot(va, showmeans=True)
plt.title('Volatile acidity')
plt.subplot(343)
plt.violinplot(ca, showmeans=True)
plt.title('Citric acid')
plt.subplot(344)
plt.violinplot(rs, showmeans=True)
plt.title("Residual sugar")
plt.subplot(345)
plt.violinplot(ch, showmeans=True)
plt.title('Chlorides')
plt.subplot(346)
plt.violinplot(fsc, showmeans=True)
plt.title("Free sulfur dioxide")
plt.subplot(347)
plt.violinplot(fsd, showmeans=True)
plt.title('Total sulfur dioxide')
plt.subplot(348)
plt.violinplot(de, showmeans=True)
plt.title('Density')
plt.subplot(349)
plt.violinplot(ph, showmeans=True)
plt.title('Ph')
plt.subplot(3, 4, 10)
plt.violinplot(su, showmeans=True)
plt.title('Sulphates')
plt.subplot(3,4, 11)
plt.violinplot(al, showmeans=True)
plt.title('Alcohol')
plt.subplot(3,4, 12)
plt.violinplot(qu, showmeans=True)
plt.title('Quality')
plt.show()
#Z-score density 
sub = df_z.copy()
for i in range(11):
    data = df_z[cols[i]].values
    for j in range(len(data)):
        k = data[j] - qu_z[j]
        sub.iloc[j, i] = k
sub.drop('quality',axis=1,inplace=True)
sns.set(font_scale=1.8)
sns.displot(data=sub, palette='Paired', kind='kde')
plt.show()