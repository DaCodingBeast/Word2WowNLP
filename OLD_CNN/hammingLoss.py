from sklearn.metrics import hamming_loss

y_true = [
    [1, 0, 1],
    [0, 1, 0]
]

y_pred = [
    [1, 1, 0],
    [0, 0, 1]
]

loss = hamming_loss(y_true, y_pred)
print(f"Hamming Loss: {loss:.2f}")
