import numpy as np
import pandas as pd

# Load the Titanic dataset (assuming 'titanic.csv' is the file name and it's in the current directory)
data = pd.read_csv('titanic-Dataset.csv')

# Preprocess the dataset
# Assuming 'Name', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', and 'Fare' are relevant features
# Encoding 'Sex' as binary (0 for male, 1 for female)
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data = data[['Name', 'Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']].dropna()

# Normalize the numerical data
for col in ['Age', 'Fare']:
    data[col] = (data[col] - data[col].mean()) / data[col].std()

# Separate features and target variable
X = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']].values
y = data['Survived'].values

# Add intercept term to feature set
X = np.hstack((np.ones((X.shape[0], 1)), X))

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Cost function
def cost_function(X, y, weights):
    m = X.shape[0]
    predictions = sigmoid(X.dot(weights))
    cost = -np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions)) / m
    return cost

# Gradient descent
def gradient_descent(X, y, weights, lr, iterations):
    m = X.shape[0]
    for _ in range(iterations):
        predictions = sigmoid(X.dot(weights))
        weights -= lr * (X.T.dot(predictions - y)) / m
    return weights

# Initialize weights
weights = np.zeros(X.shape[1])

# Train the model
lr = 0.001
iterations = 90000
weights = gradient_descent(X, y, weights, lr, iterations)

# Predict function
def predict(X, weights):
    X = np.hstack((np.ones((X.shape[0], 1)), X))  # Add intercept term
    return sigmoid(X.dot(weights)) >= 0.5

# Generate random data points and predict
np.random.seed(42)  # For reproducibility
random_data = np.random.rand(5, 6)  # Generate 5 random data points with 6 features each
random_names = ["John Doe", "Jane Smith", "Alice Brown", "Bob White", "Charlie Green"]
random_predictions = predict(random_data, weights)

# Output predictions with random names
print("Predictions on random data points:")
for name, pred in zip(random_data,random_predictions):
    status = 'Survived' if pred else 'Did not survive'
    print(f"{name[0]}: {status}")