import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree


df = pd.read_csv('train_data_tree.csv')
clf = tree.DecisionTreeClassifier(criterion='entropy')
X = df[['sex', 'exang']]
y = df['num']
clf.fit(X, y)
tree.plot_tree(clf, filled=True)

plt.show()