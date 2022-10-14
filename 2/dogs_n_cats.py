import pandas as pd
from sklearn import tree

df_train = pd.read_csv('dogs_n_cats.csv')
df_test = pd.read_json('dataset_209691_15.txt')

X_train = df_train[df_train.columns[0:5]]
X_test = df_test

y_train = df_train['Вид']

clf = tree.DecisionTreeClassifier(criterion='entropy')
clf.fit(X_train, y_train)

result = pd.Series(clf.predict(X_test))

print(df_test)
print(result)
print(result[result == 'собачка'].count())
