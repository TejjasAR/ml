import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.datasets import load_breast_cancer


X = load_breast_cancer().data


X = StandardScaler().fit_transform(X)
X_pca = PCA(2).fit_transform(X)


y = KMeans(2).fit_predict(X)


plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
plt.colorbar(label="Cluster Label")

plt.suptitle("K-Means Clustering on Breast Cancer Dataset")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")

plt.grid()
plt.show()