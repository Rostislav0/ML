import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import collections
import matplotlib.pyplot as plt

df_train = pd.read_csv('invasion.csv')
df_test = pd.read_csv('operative_information.csv')

X_train = df_train.drop('class', axis=1)
X_test = df_test

# transport = {'transport': 0, 'fighter': 1, 'cruiser': 2}
# y_train = df_train['class'].map(transport)

y_train = df_train['class']

clf = RandomForestClassifier(criterion='entropy', random_state=0)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print(pd.Series(predictions).value_counts())

# # or

# transport = {0: 'transport', 1: 'fighter', 2: 'cruiser'}
# for i, k in collections.Counter(predictions).items():
#     print(transport[i], k)


# important columns
imp = pd.DataFrame(clf.feature_importances_, index=X_train.columns, columns=['importance'])
imp.sort_values('importance').plot(kind='barh', figsize=(12, 8))
plt.show()
