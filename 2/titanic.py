import pandas as pd
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split, cross_val_score
import seaborn as sns

titanic_data = pd.read_csv('titanic (1).csv')

X = titanic_data.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'], axis=1)
y = titanic_data.Survived
X = pd.get_dummies(X)
X = X.fillna({'Age': X.Age.median()})
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

data = []
for max_depth in range(1, 100):
    clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=max_depth)
    clf.fit(X_train, y_train)
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)

    mean_cr_val_scor = cross_val_score(clf, X_train, y_train, cv=5).mean()
    temp_data = (max_depth, train_score, test_score, mean_cr_val_scor)
    data.append(temp_data)
scores_data = pd.DataFrame(data, columns=['max_depth', 'train_score', 'test_score', 'cross_val_score'])
scores_data_long = pd.melt(scores_data, id_vars=['max_depth'],
                           value_vars=['train_score', 'test_score', 'cross_val_score'],
                           var_name='set_type', value_name='score')

sns.lineplot(x="max_depth", y='score', hue='set_type', data=scores_data_long)
best_clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=8)
# print(scores_data_long.query("set_type == 'cross_val_score'").head(20))
best_clf.fit(X_train, y_train)

print(best_clf.score(X_test, y_test))
plt.show()
