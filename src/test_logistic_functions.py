
import numpy as np
from logistic import sigmoid, compute_cost, gradient_descent, predict

# Small artificial dataset for checking the implementation
X = np.array([
    [-2.0],
    [-1.0],
    [1.0],
    [2.0]
])

y = np.array([0, 0, 1, 1])

# Check sigmoid
print("Sigmoid test:", sigmoid(np.array([-2.0, 0.0, 2.0])))

# Check initial cost when all weights are zero
theta_initial = np.zeros(X.shape[1] + 1)
print("Initial cost:", compute_cost(X, y, theta_initial))

# Train on the small artificial dataset
theta, history = gradient_descent(
    X, y, alpha=0.1, iterations=2000
)

print("Learned theta:", theta)
print("Initial recorded cost:", history[0])
print("Final cost:", history[-1])
print("Predictions:", predict(X, theta))
print("Actual labels:", y)
