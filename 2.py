import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing

df = fetch_california_housing(as_frame=True).frame
print(df.head())    


corr = df.corr()
print("\nCorrelation Matrix:")
print(corr)

sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix Heatmap")



sns.pairplot(df, diag_kind='kde',corner = True)
plt.show()