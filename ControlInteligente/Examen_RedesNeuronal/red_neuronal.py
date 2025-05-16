

import funciones as fn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


"""
Maestria en ciencias - 16 - 05 -2025

P | x1 | x2 | x3 | x4 | x5 | y |
0 | 0  | 0  | 0 | 0  | 0 | 0 |
1 | 0  | 0  | 0 | 0  | 1 | 1 |
2 | 0  | 0  | 0 | 1  | 0 | 2 |
3 | 0  | 0  | 0 | 1  | 1 | 3 |
4 | 0  | 0  | 1 | 0  | 0 | 4 |
5 | 0  | 0  | 1 | 0  | 1 | 5 |
6 | 0  | 0  | 1 | 1  | 0 | 6 |
7 | 0  | 0  | 1 | 1  | 1 | 7 |
8 | 0  | 1  | 0 | 0  | 0 | 8 |
9 | 0  | 1  | 0 | 0  | 1 | 9 |

Primera capa: 5 entradas -> 4 neuronas 
Segunda capa: 4 entradas -> 1 salida
Funcion de activacion: 
Error: MSE
"""

def forward(X, w1, b1, w2, b2):
    z1 = np.dot(X, w1) + b1
    a1 = fn.relu(z1)
    z2 = np.dot(a1, w2) + b2
    a2 = fn.linear(z2)
    return a1, a2

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def backward(X, y, a1, a2, w2, b2):
    m = y.shape[0]
    dz2 = a2 - y
    dw2 = np.dot(a1.T, dz2) / m
    db2 = np.sum(dz2, axis=0) / m
    da1 = np.dot(dz2, w2.T)
    dz1 = da1 * fn.relu_deriv(a1)
    dw1 = np.dot(X.T, dz1) / m
    db1 = np.sum(dz1, axis=0) / m
    return dw1, db1, dw2, db2

def update_weights(w1, b1, w2, b2, dw1, db1, dw2, db2, learning_rate):
    w1 -= learning_rate * dw1
    b1 -= learning_rate * db1
    w2 -= learning_rate * dw2
    b2 -= learning_rate * db2
    return w1, b1, w2, b2

def train(X, y, w1, b1, w2, b2, epochs, learning_rate, error_threshold):
    loss_history = []
    epoch = 0
    while epoch < epochs:
        a1, a2 = forward(X, w1, b1, w2, b2)
        loss = mse(y, a2)
        loss_history.append(loss)
        if loss <= error_threshold:
            print(f"Early stop at epoch {epoch}, Loss: {loss}")
            break
        dw1, db1, dw2, db2 = backward(X, y, a1, a2, w2, b2)
        w1, b1, w2, b2 = update_weights(w1, b1, w2, b2, dw1, db1, dw2, db2, learning_rate)
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")
        epoch += 1
    return w1, b1, w2, b2, loss_history

def predict(X, w1, b1, w2, b2):
    _, a2 = forward(X, w1, b1, w2, b2)
    return a2 

def plot_loss(loss_history):
    plt.plot(loss_history)
    plt.title("Loss History")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

def main():

    X = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 1, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 1],
        [0, 0, 1, 1, 0],
        [0, 0, 1, 1, 1],
        [0, 1, 0, 0, 0],
        [0, 1, 0, 0, 1]
    ])

    y = np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]])

    w1 = np.random.rand(5, 4)
    b1 = np.random.rand(4)
    w2 = np.random.rand(4, 1)
    b2 = np.random.rand(1)

    epochs = 100000
    learning_rate = 0.01
    error_threshold = 0.001
    w1, b1, w2, b2, loss_history = train(X, y, w1, b1, w2, b2, epochs, learning_rate, error_threshold)
    a1, a2 = forward(X, w1, b1, w2, b2)
    loss = mse(y, a2)
    print(f"Final Loss: {loss}")
    predictions = predict(X, w1, b1, w2, b2)
    print("MSE:", mse(y, predictions))
    results_df = pd.DataFrame(np.hstack((X, predictions, y)), columns=["x1", "x2", "x3", "x4", "x5", "Predicted", "Actual"])
    print("\nResults Table:")
    print(results_df)
    plot_loss(loss_history)

if __name__ == "__main__":
    main()