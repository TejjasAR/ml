import numpy as np
from sklearn.neighbors import KNeighborsClassifier

x = np.random.rand(100)

X_train = x[:50].reshape(-1, 1)
y_train = np.where(x[:50] <= 0.5, 1, 2)
X_test = x[50:].reshape(-1, 1)

print("k-NN Classification Results:")

for k in [1, 2, 3, 4, 5, 20, 30]:
    model = KNeighborsClassifier(k).fit(X_train, y_train)

    print("\nk =", k)
    print(model.predict(X_test))