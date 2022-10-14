import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt

all_data = pd.read_csv('forest_dataset.csv')

labels = all_data[all_data.columns[-1]].values
feature_matrix = all_data[all_data.columns[:-1]].values

X_train, X_test, y_train, y_test = train_test_split(
    feature_matrix, labels, test_size=0.2, random_state=42)

clf = KNeighborsClassifier()

params = {'n_neighbors': range(1, 11),
          'metric': ['manhattan', 'euclidean'],
          'weights': ['uniform', 'distance']}

clf_grid = GridSearchCV(clf, params, cv=5, scoring='accuracy', n_jobs=-1)
clf_grid.fit(X_train, y_train)
y_pred = clf_grid.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(clf_grid.best_params_)

unique, freq = np.unique(y_test, return_counts=True)
freq = list(map(lambda x: x / len(y_test), freq))

pred_prob = clf_grid.predict_proba(X_test)

pred_freq = pred_prob.mean(axis=0)
plt.figure(figsize=(10, 8))
plt.bar(range(1, 8), pred_freq, width=0.4, align="edge", label='prediction')
plt.bar(range(1, 8), freq, width=-0.4, align="edge", label='real')
plt.ylim(0, 0.54)
plt.legend()
plt.show()
print(freq)
