import numpy as np

A = np.array([1, 2, 3, 4, 5, -1, -2, -3, -4, -5])
B = np.array([0, 2, 2, 5, 3, -1, -1, -4, -6, -5])

MAE = 1 / len(B) * np.sum(np.abs(A - B))
print(MAE)
MAPE = 100 / len(B) * np.sum(np.abs((A - B) / A))
print(MAPE)
