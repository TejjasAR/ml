
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

df = fetch_california_housing(as_frame=True).frame

print("Dataset Sample:")
print(df.head())

df.hist(figsize=(12,10), bins=30)
plt.suptitle("Histograms of Numerical Features")
plt.show()



plt.figure(figsize=(14, 10))
for i, col in enumerate(df.columns, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(y=df[col])
    plt.title(col)
plt.show()


for col in df.columns:
    mean=round(df[col].mean(), 2)
    median=round(df[col].median(), 2)
    std_dev=round(df[col].std(), 2)

    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    outliers = ((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum()

    print("\nFeature:",col)
    print("Mean:", mean)
    print("Median:", median)
    print("Std Dev:", std_dev)
    print("Number of Outliers:", outliers)