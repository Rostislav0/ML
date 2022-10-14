import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv('space_can_be_a_dangerous_place.csv')

sns.heatmap(df.corr())
plt.show()