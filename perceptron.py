import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate=learning_rate
        self.num_iterations=num_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.num_iterations):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self._step_function(linear_output)

                update = self.learning_rate * (y[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self._step_function(linear_output)
        return y_predicted

    def _step_function(self, x):
        return np.where(x>=0,1,0)


#Example:
if __name__ == "__main__":
    X = np.array([[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]])
    y = np.array([0, 0, 0, 1, 1, 1])

    perceptron = Perceptron()
    perceptron.fit(X, y)

    predictions = perceptron.predict(X)
    print(predictions) 

    #Visualization
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', label='Actual')

    #Plotting
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x_values = np.linspace(x_min, x_max, 100)
    y_values = -(perceptron.weights[0] * x_values + perceptron.bias) / perceptron.weights[1]

    plt.plot(x_values, y_values, color='red', label='Decision Boundary')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Perceptron')
    plt.legend()
    plt.show()