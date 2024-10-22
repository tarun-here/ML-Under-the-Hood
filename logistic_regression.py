import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        #Gradient descent
        for _ in range(self.num_iterations):
            linear_model = np.dot(X, self.weights) + self.bias

            #sigmoid function
            y_predicted = self._sigmoid(linear_model)

            #gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            #weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)


#Example:
if __name__ == "__main__":
    #data
    X=np.array([[1],[2],[3],[4],[5]])
    y=np.array([0,0,1,1,1])

    #model
    model =LogisticRegression()
    model.fit(X, y)

    #predictions
    predictions = model.predict(X)
    print(predictions)

    # Visualization
    plt.figure(figsize=(8, 6))
    plt.scatter(X,y,color='blue',label='Actual')

    #Plotting
    X_test =np.linspace(0,6,100)
    y_pred_prob =model._sigmoid(model.weights*X_test +model.bias)
    plt.plot(X_test,y_pred_prob,color='red',label='Decision Boundary')

    plt.xlabel('X')
    plt.ylabel('y (Probability)')
    plt.title('Logistic Regression')
    plt.legend()
    plt.show()