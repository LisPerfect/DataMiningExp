from sklearn.feature_selection import SelectKBest, f_regression
from pandas import read_csv
df = read_csv('./after_preprocessing.csv')
y_train = df['quality']
x = df.drop('quality', axis=1)
X_train = x
selector = SelectKBest(f_regression, k=5)#V
X_new = selector.fit_transform(X_train, y_train)
selected_feature_indices = selector.get_support(indices=True)
for i in selected_feature_indices:
    print(df.columns[i])