import numpy as np
import scipy.special as ss

# Outliers
A = np.array([1, 4, 10, 2, 6, 3, 50])

print(A[ss.erfc(abs(A - np.average(A)) / np.std(A)) > 1 / (len(A) * 2)])
print(ss.erfc(abs(A - 10) / 1.1) < 1 / (10 * 2))


# Sum distances between points
A = np.array([1, 1, 0])
B = np.array([0, 2, -1])
C = np.array([2, 3, 1])
D = np.array([1, 0, 4])
l = {'A': A, 'B': B, 'C': C, 'D': D}
for i1, i2 in l.items():
    k = 0
    for a in l.values():
        k += np.linalg.norm(i2 - a, ord=1)
    print(i1, k)
