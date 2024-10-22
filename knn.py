import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    def _predict(self, x):
        #distances
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        #k nearest samples, labels
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

#Example:
if __name__ == "__main__":
    #data 
    X = np.array([[1, 1], [1.5, 2], [3, 4], [5, 6], [3, 1], [4.5, 5]])
    y = np.array([0, 0, 1, 1, 0, 1]) 

    #model
    k = 3 
    model = KNN(k=k)
    model.fit(X, y)

    #predictions
    predictions = model.predict(X)
    print(predictions)

    #Visualization
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', label='Actual')

    #Plotting
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title(f'KNN (k={k})')
    plt.legend()
    plt.show()