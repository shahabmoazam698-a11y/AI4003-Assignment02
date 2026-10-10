
import numpy as np


def sigmoid(z):
    """Convert scores into probabilities between 0 and 1."""
    # Clip values to avoid numerical overflow in exp()
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def add_intercept(X):
    """Add a column of ones for the intercept parameter theta_0."""
    return np.column_stack((np.ones(X.shape[0]), X))


def predict_proba(X, theta):
    """Calculate predicted probabilities."""
    X_with_intercept = add_intercept(X)
    return sigmoid(X_with_intercept @ theta)


def compute_cost(X, y, theta):
    """Calculate binary cross-entropy cost."""
    probabilities = predict_proba(X, theta)

    # Avoid log(0)
    probabilities = np.clip(
        probabilities, 1e-12, 1.0 - 1e-12
    )

    m = len(y)

    cost = -np.mean(
        y * np.log(probabilities)
        + (1 - y) * np.log(1 - probabilities)
    )

    return cost


def gradient_descent(X, y, alpha, iterations=2000):
    """Train logistic regression using batch gradient descent."""
    X_with_intercept = add_intercept(X)

    m, n = X_with_intercept.shape
    theta = np.zeros(n)

    cost_history = []

    for _ in range(iterations):
        probabilities = sigmoid(X_with_intercept @ theta)

        # Gradient of binary cross-entropy cost
        gradient = (X_with_intercept.T @ (probabilities - y)) / m

        # Update all parameters simultaneously
        theta = theta - alpha * gradient

        # Store cost after the update
        cost_history.append(
            compute_cost(X, y, theta)
        )

    return theta, np.array(cost_history)


def predict(X, theta, threshold=0.5):
    """Convert predicted probabilities into class labels."""
    probabilities = predict_proba(X, theta)
    return (probabilities >= threshold).astype(int)
