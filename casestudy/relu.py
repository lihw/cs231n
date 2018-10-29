import numpy as np
import matplotlib.pyplot as plt


N = 100 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes
X = np.zeros((N*K,D)) # data matrix (each row = single example)
y = np.zeros(N*K, dtype='uint8') # class labels
for j in range(K):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
  y[ix] = j
# lets visualize the data:
#plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
#plt.show()

# Initialize W and b
h = 100
W = 0.01 * np.random.randn(D, h)
b = np.zeros((1, h))
W2 = 0.01 * np.random.randn(h, K)
b2 = np.zeros((1, K))

num_examples = X.shape[0]
step_size = 1e-0
reg = 1e-3

for i in range(10000):
    hidden_layer = np.maximum(0, np.dot(X, W) + b) # relu
    scores = np.dot(hidden_layer, W2) + b2
    
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis = 1, keepdims = True)

    correct_probs = -np.log(probs[range(num_examples), y])

    data_loss = np.sum(correct_probs) / num_examples
    reg_loss = 0.5 * reg * np.sum(W * W) + 0.5 * reg * np.sum(W2 * W2)
    loss = data_loss + reg_loss
    if i % 1000 == 0:
        print("iteration %d: loss %f" % (i, loss))

    dscores = probs
    dscores[range(num_examples), y] -= 1
    dscores /= num_examples

    dW2 = np.dot(hidden_layer.T, dscores)
    dW2 += reg * W2
    db2 = np.sum(dscores, axis = 0, keepdims = True)

    dhidden_layer = np.dot(dscores, W2.T)
    # backprop the ReLU non-linearity
    dhidden_layer[hidden_layer <= 0] = 0

    dW = np.dot(X.T, dhidden_layer)
    dW += reg * W
    db = np.sum(dhidden_layer, axis = 0, keepdims = True)

    W2 += dW2 * -step_size
    b2 += db2 * -step_size

    W += dW * -step_size
    b += db * -step_size

# evaluate training set accuracy
hidden_layer = np.maximum(0, np.dot(X, W) + b) # relu
scores = np.dot(hidden_layer, W2) + b2
predicted_class = np.argmax(scores, axis=1)
print('training accuracy: %.2f' % (np.mean(predicted_class == y)))

plt.scatter(X[:, 0], X[:, 1], c=predicted_class, s=40, cmap=plt.cm.Spectral)
plt.show()
