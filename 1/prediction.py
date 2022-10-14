from sklearn.linear_model import Ridge
import numpy as np

X = np.array([[0, 3],
              [1, 2],
              [2, 1],
              [3, 0]])
# Добавляем дополнительный фиктивный
# столбец для свободного члена
X = np.insert(X, 0, values=1, axis=1)
Y = np.array([0, 1, 0, 3])
reg = Ridge(alpha=1.0, fit_intercept=False)
reg.fit(X, Y)
print(reg.coef_)
