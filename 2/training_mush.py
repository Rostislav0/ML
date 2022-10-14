import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import seaborn as sns

params = {'n_estimators': range(10, 50, 10),
          'max_depth': range(1, 12, 2),
          'min_samples_leaf': range(1, 7),
          'min_samples_split': range(2, 9, 2)}

df_train = pd.read_csv('training_mush.csv')
df_test = pd.read_csv('testing_mush.csv')
df_test_y = pd.read_csv('testing_y_mush.csv')

X_train = df_train.drop('class', axis=1)
y_train = df_train['class']
clf = RandomForestClassifier(criterion='entropy', random_state=0)

# # get the best params
# search = GridSearchCV(clf, params, cv=3, n_jobs=-1)
# search.fit(X_train, y_train)
# best_params = search.best_params_
# print(best_params)

# # important of columns
# clf.fit(X_train, y_train)
# imp = pd.DataFrame(clf.feature_importances_, index=X_train.columns, columns=['importance'])
# imp.sort_values('importance').plot(kind='barh', figsize=(12, 8))
# plt.show()

# amount predict values equal 1
clf.fit(X_train, y_train)
predicts = pd.DataFrame(clf.predict(df_test)).rename({0: 'predict_values'}, axis=1)
print(np.sum(predicts['predict_values'] == 1))

# Confusion matrix for predict values and real values
diff = sklearn.metrics.confusion_matrix(df_test_y, predicts)
sns.heatmap(diff, annot=True, annot_kws={"size": 16})
plt.show()