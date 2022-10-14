import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import tree
import seaborn as sns
np.random.seed(0)
rs = np.random.seed(0)

train_iris = pd.read_csv('train_iris.csv')
test_iris = pd.read_csv('test_iris.csv')


X_train = train_iris[train_iris.columns[1:5]]
X_test = test_iris[train_iris.columns[1:5]]

y_train = train_iris.species
y_test = test_iris.species

data = []
for max_depth in range(1, 100):
    clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=max_depth)
    clf.fit(X_train, y_train)
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)

    # mean_cr_val_scor = cross_val_score(clf, X_train, y_train, cv=5).mean()
    temp_data = (max_depth, train_score, test_score)
    data.append(temp_data)

scores_data = pd.DataFrame(data, columns=['max_depth', 'train_score', 'test_score'])
scores_data_long = pd.melt(scores_data, id_vars=['max_depth'],
                           value_vars=['train_score', 'test_score'],
                           var_name='set_type', value_name='score')
sns.lineplot(x="max_depth", y='score', hue='set_type', data=scores_data_long)
plt.show()
