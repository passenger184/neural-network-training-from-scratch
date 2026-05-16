import numpy as np

np.random.seed(42)

# ── Setup (same as before) ──
batch_size   = 5
input_size   = 5
hidden1_size = 6
hidden2_size = 8
hidden3_size = 8
output_size  = 3

true_labels = np.array([0, 2, 1, 0, 2])
X = np.random.randn(batch_size, input_size)

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy_loss(probs, labels):
    batch_size = probs.shape[0]
    correct = probs[np.arange(batch_size), labels]
    return -np.mean(np.log(correct + 1e-8))

# ── Initialize weights ──
W1 = np.random.randn(input_size,   hidden1_size) * 0.1
b1 = np.zeros(hidden1_size)
W2 = np.random.randn(hidden1_size, hidden2_size) * 0.1
b2 = np.zeros(hidden2_size)
W3 = np.random.randn(hidden2_size, hidden3_size) * 0.1
b3 = np.zeros(hidden3_size)
W4 = np.random.randn(hidden3_size, output_size)  * 0.1
b4 = np.zeros(output_size)

learning_rate = 0.4
epochs        = 100
loss_history  = []

print("Starting training...\n")

for epoch in range(epochs):

    # FORWARD PASS

    # Layer 1: linear transformation then ReLU
    Z1 = X @ W1 + b1              # shape: (5,6)
    A1 = relu(Z1)                 # shape: (5,6) — negatives killed

    # Layer 2: linear transformation then ReLU
    Z2 = A1 @ W2 + b2             # shape: (5,8)
    A2 = relu(Z2)                 # shape: (5,8)

    # Layer 3: linear transformation then ReLU
    Z3 = A2 @ W3 + b3             # shape: (5,8)
    A3 = relu(Z3)                 # shape: (5,8)

    # Output layer: linear transformation then softmax
    Z4 = A3 @ W4 + b4             # shape: (5,3) — raw scores (logits)
    probs = softmax(Z4)           # shape: (5,3) — probabilities, each row sums to 1

    # Calculate loss — one number telling us how wrong we are
    loss = cross_entropy_loss(probs, true_labels)
    loss_history.append(loss)

    # Calculate accuracy — how many did we get right
    predicted = np.argmax(probs, axis=1)          # index of highest prob per sample
    accuracy  = np.mean(predicted == true_labels)

    # Print every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d} | Loss: {loss:.4f} | Accuracy: {accuracy:.0%}")

    # BACKWARD PASS
    one_hot = np.zeros_like(probs)                       # (5,3) all zeros
    one_hot[np.arange(batch_size), true_labels] = 1      # put 1 in correct class position

    dZ4 = (probs - one_hot) / batch_size      
    dW4 = A3.T @ dZ4                          
    db4 = dZ4.sum(axis=0)                     
    dA3 = dZ4 @ W4.T                           
    dZ3 = dA3 * (Z3 > 0)                    
  
    # Repeat same pattern for layer 3 → layer 2
    dW3 = A2.T @ dZ3                           # shape: (8,8)
    db3 = dZ3.sum(axis=0)                      # shape: (8,)
    dA2 = dZ3 @ W3.T                           # shape: (5,8)
    dZ2 = dA2 * (Z2 > 0)                       # ReLU gradient

    # Repeat same pattern for layer 2 → layer 1
    dW2 = A1.T @ dZ2                           # shape: (6,8)
    db2 = dZ2.sum(axis=0)                      # shape: (8,)
    dA1 = dZ2 @ W2.T                           # shape: (5,6)
    dZ1 = dA1 * (Z1 > 0)                       # ReLU gradient

    # Repeat same pattern for layer 1 → input
    dW1 = X.T @ dZ1                            # shape: (5,6)
    db1 = dZ1.sum(axis=0)                      # shape: (6,)

    # WEIGHT UPDATE

    W4 -= learning_rate * dW4
    b4 -= learning_rate * db4
    W3 -= learning_rate * dW3
    b3 -= learning_rate * db3
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

# ── Final Results ──
print(f"\nStarting loss: {loss_history[0]:.4f}")
print(f"Final loss:    {loss_history[-1]:.4f}")
print(f"Loss went down by: {loss_history[0] - loss_history[-1]:.4f}")
