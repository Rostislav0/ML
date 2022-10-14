from sklearn.metrics import roc_auc_score


print(roc_auc_score([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1], [0, 0.45, 0.4, 0.76, 0.5, 0.65, 0.7, 0.75, 0.3, 0.65, 0.7, 0.76, 0.6, 0.8, 0.71, 0.7]))