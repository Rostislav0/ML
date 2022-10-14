import numpy as np


g = np.array([[15, 30],
              [15, 40]])
accuracy = np.sum(np.diag(g))/np.sum(g)
print(np.around(accuracy, 2))
precision = np.diag(g)[-1]/np.sum(g[-1])
print(np.around(precision, 2))
recall = np.diag(g)[-1]/sum(g)[-1]
print(np.around(recall, 2))

#
# g = np.array([[0, 0],
#               [20, 30]])
# accuracy = np.sum(np.diag(g))/np.sum(g)
# print(np.around(accuracy, 2))
# precision = np.diag(g)[-1]/np.sum(g[-1])
# print(np.around(precision, 2))
# recall = np.diag(g)[-1]/sum(g)[-1]
# print(np.around(recall, 2))