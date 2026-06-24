import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)

model = DecisionTreeClassifier().fit(X_train, y_train)

plt.figure(figsize=(16, 10))
plot_tree(model,filled=True)
plt.show()

sample = X_test[0].reshape(1, -1)
print("new sample class:",data.target_names[model.predict(sample)[0]])