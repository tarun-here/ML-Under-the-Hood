import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        #Gradient descent
        for _ in range(self.num_iterations):
            y_predicted = np.dot(X, self.weights) + self.bias

            dw=(1/n_samples)*np.dot(X.T,(y_predicted-y))
            db=(1/n_samples)*np.sum(y_predicted-y)

            # Update weights and bias
            self.weights-=self.learning_rate*dw
            self.bias-=self.learning_rate*db

    def predict(self,X):
        return np.dot(X,self.weights)+ self.bias


#Example:
if __name__ =="__main__":
    X =np.array([[1], [2], [3], [4], [5]])
    y =np.array([2,4,5,4,5])

    #model
    model = LinearRegression()
    model.fit(X, y)

    #predictions
    predictions = model.predict(X)
    print(predictions)
    plt.scatter(X, y, color='blue', label='Actual')
    #Plotting
    plt.plot(X, predictions, color='red', label='Predicted')

    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Linear Regression')
    plt.legend()
    plt.show()