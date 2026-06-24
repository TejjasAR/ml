import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris


iris = load_iris()

X = iris.data    
y = iris.target   

X = StandardScaler().fit_transform(X)        
X_pca = PCA(2).fit_transform(X)    


plt.scatter(X_pca[:,0],X_pca[:,1],c=y)


plt.suptitle("PCA of Iris Dataset (2 Components)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.grid()  
plt.show()