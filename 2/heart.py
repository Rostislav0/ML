import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier


np.random.seed(0)

df = pd.read_csv('heart.csv')

X_train = df.drop('target', axis=1)
y_train = df.target

rf = RandomForestClassifier(10, max_depth=5)
rf.fit(X_train, y_train)

imp = pd.DataFrame(rf.feature_importances_, index=X_train.columns, columns=['importance'])
imp.sort_values('importance').plot(kind='barh', figsize=(12, 8))
plt.show()
