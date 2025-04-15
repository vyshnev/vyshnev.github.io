# As background visualizer we train a small NN on synthetic 2D classification data
#
import numpy as np
import itertools
from pyodide.ffi import to_js
# from js import drawPoints, document # Not needed if drawing is purely JS

def generate():
    """Generates synthetic 2D blob data for binary classification."""
    global x, y  # Make x and y accessible globally within this script
    n = 1000
    x = np.random.rand(n, 2)
    y = np.zeros(n)
    # The center of a next blob should be within a certain distance of the previous one
    n_blobs = 10
    center = None # Initialize center
    radius = 0.1 # Initialize radius

    for i in range(n_blobs):
        if i == 0:
            # For first blob, place it near the center
            center = np.random.rand(2) * 0.5 + 0.25
        else:
            # Subsequent blobs are placed relative to the previous center
             # Ensure center is defined before trying to use it
            if center is not None:
                 center = center + np.random.rand(2) * 3 * radius - radius
                 # Clamp center coordinates to be within [0, 1]
                 center = np.clip(center, 0, 1)
            else:
                 # Fallback if center is somehow None (shouldn't happen after i=0)
                 center = np.random.rand(2) * 0.5 + 0.25

        radius = np.random.rand() * 0.2 + 0.05 # Adjust radius range if needed (0.05 to 0.25)

        # Add blob influence to y
        if center is not None:
            y += np.exp(-np.sum((x - center)**2, axis=1) / radius**2)

    # Normalize y based on its distribution (e.g., using median or a threshold)
    threshold = np.median(y[y > 0]) if np.any(y > 0) else 0.5 # Use median of positive values
    y = (y > threshold).astype(int)

    # Put into one array with 3 columns [x1, x2, label]
    data = np.hstack([x, y.reshape(-1, 1)])
    return to_js(data.tolist()) # Convert NumPy array to list, then to JS object

class Sigmoid():
    """Sigmoid activation function."""
    def __init__(self):
        self.y = None # Store output for backward pass

    def forward(self, x):
        y = 1 / (1 + np.exp(-np.clip(x, -500, 500))) # Clip to avoid overflow
        self.y = y
        return y

    def backward(self, dLdy, opt):
        # Calculate gradient dL/dx using chain rule: dL/dx = dL/dy * dy/dx
        # dy/dx for sigmoid is y * (1 - y)
        return dLdy * self.y * (1 - self.y)

class ReLU():
    """ReLU activation function."""
    def __init__(self):
        self.x = None # Store input for backward pass

    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, dLdy, opt):
        # Gradient is dL/dy where x > 0, and 0 otherwise
        mask = self.x > 0
        return dLdy * mask

class GELU():
    """GELU activation function (approximation)."""
    def __init__(self):
        self.x = None # Store input for backward pass

    def forward(self, x):
        self.x = x
        # Using the common GELU approximation
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

    def backward(self, dLdy, opt):
        x = self.x
        # Derivative of the GELU approximation
        # Let k = sqrt(2 / pi) * (x + 0.044715 * x^3)
        # GELU_approx = 0.5 * x * (1 + tanh(k))
        # d(GELU_approx)/dx = 0.5 * (1 + tanh(k)) + 0.5 * x * sech^2(k) * d(k)/dx
        # d(k)/dx = sqrt(2 / pi) * (1 + 3 * 0.044715 * x^2)

        k_const = np.sqrt(2 / np.pi)
        k = k_const * (x + 0.044715 * x ** 3)
        tanh_k = np.tanh(k)
        sech_k_sq = 1 - tanh_k**2 # sech^2(x) = 1 - tanh^2(x)
        dk_dx = k_const * (1 + 3 * 0.044715 * x**2)

        dx = 0.5 * (1 + tanh_k) + 0.5 * x * sech_k_sq * dk_dx
        return dLdy * dx

class Linear():
    """Linear layer (fully connected)."""
    id_iter = itertools.count() # Class variable for unique IDs
    def __init__(self, in_size, out_size):
        self.x = None # Store input for backward pass
        self.id_w = next(Linear.id_iter) # Unique ID for weights
        self.id_b = next(Linear.id_iter) # Unique ID for bias
        # Xavier/Glorot initialization (good for tanh/sigmoid like activations)
        # limit = np.sqrt(6 / (in_size + out_size))
        # He initialization (good for ReLU/GELU)
        limit = np.sqrt(2 / in_size)
        self.weights = np.random.uniform(-limit, limit, (in_size, out_size))
        # self.bias = np.zeros(out_size) # Initialize bias to zero often works well
        self.bias = np.random.uniform(-limit, limit, out_size) # Or initialize bias similarly


    def forward(self, x):
        self.x = x
        # y = x @ W + b
        return x @ self.weights + self.bias

    def backward(self, dLdy, opt):
        # Calculate gradients dL/dw, dL/db, dL/dx
        # dL/dw = x^T @ dL/dy
        dLdw = self.x.T @ dLdy
        # dL/db = sum(dL/dy) over batch dimension
        dLdb = np.sum(dLdy, axis=0)
        # dL/dx = dL/dy @ W^T
        dLdx = dLdy @ self.weights.T

        # Update weights and bias using the optimizer
        self.weights = opt.update(self.id_w, self.weights, dLdw)
        self.bias = opt.update(self.id_b, self.bias, dLdb)

        return dLdx # Pass gradient back to the previous layer

class NeuralNetwork():
    """Simple feedforward neural network."""
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dLdy, opt):
        for layer in reversed(self.layers): # Go backwards through layers
            dLdy = layer.backward(dLdy, opt)
        return dLdy # Final gradient w.r.t input (not usually used)

class BinaryCrossEntropy():
    """Binary Cross Entropy loss function."""
    def __init__(self, epsilon=1e-12): # Add epsilon for numerical stability
        self.y_pred = None # Predicted probabilities
        self.y_true = None # True labels (0 or 1)
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        # Ensure shapes are compatible
        y_true = y_true.reshape(-1, 1) if len(y_true.shape) == 1 else y_true
        y_pred = np.clip(y_pred, self.epsilon, 1. - self.epsilon) # Clip predictions

        self.y_pred = y_pred
        self.y_true = y_true

        # Calculate BCE loss: -mean(t*log(y) + (1-t)*log(1-y))
        loss = -(self.y_true * np.log(self.y_pred) + (1 - self.y_true) * np.log(1 - self.y_pred))
        return np.mean(loss)

    def backward(self):
        # Gradient of BCE loss w.r.t y_pred: (y_pred - y_true) / (y_pred * (1 - y_pred))
        # Need to divide by batch size for mean gradient
        batch_size = self.y_true.shape[0]
        dLdy = (self.y_pred - self.y_true) / (self.y_pred * (1 - self.y_pred) * batch_size)
        return dLdy

class AdamW():
    """AdamW optimizer."""
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.01):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.m = {} # First moment estimates
        self.v = {} # Second moment estimates
        self.t = {} # Timesteps for bias correction

    def update(self, param_id, param, grad):
        # Increment timestep for the specific parameter
        self.t[param_id] = self.t.get(param_id, 0) + 1
        t = self.t[param_id]

        # Initialize moment estimates if they don't exist
        if param_id not in self.m:
            self.m[param_id] = np.zeros_like(param)
            self.v[param_id] = np.zeros_like(param)

        m_prev = self.m[param_id]
        v_prev = self.v[param_id]

        # Apply weight decay BEFORE calculating momentum (AdamW characteristic)
        param -= self.lr * self.weight_decay * param

        # Update biased first moment estimate
        m = self.beta1 * m_prev + (1 - self.beta1) * grad
        # Update biased second raw moment estimate
        v = self.beta2 * v_prev + (1 - self.beta2) * (grad ** 2)

        # Compute bias-corrected moment estimates
        m_hat = m / (1 - self.beta1 ** t)
        v_hat = v / (1 - self.beta2 ** t)

        # Update parameters
        param -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

        # Store updated moments
        self.m[param_id] = m
        self.v[param_id] = v

        return param

# --- Global variables for the model, criterion, optimizer, and iteration count ---
classifier = None
criterion = None
opt = None
iteration_count = 0

def initialize_model():
    """Initializes or re-initializes the neural network model."""
    global classifier, criterion, opt, iteration_count
    print("Initializing model...")
    hidden_size = 8
    # Randomly choose between ReLU and GELU activation
    activation_class = ReLU if np.random.rand() < 0.5 else GELU
    print(f"Using {activation_class.__name__} activation.")

    classifier = NeuralNetwork([
        Linear(2, hidden_size),     # Input (2D) to hidden
        activation_class(),
        Linear(hidden_size, hidden_size), # Hidden to hidden
        activation_class(),
        Linear(hidden_size, 1),     # Hidden to output (1D for binary)
        Sigmoid()                   # Sigmoid for binary probability output
    ])
    criterion = BinaryCrossEntropy()
    opt = AdamW(lr=0.01, weight_decay=0.0001) # Define optimizer
    iteration_count = 0 # Reset iteration count on initialization
    Linear.id_iter = itertools.count() # Reset parameter IDs

def step():
    """Performs one training step and returns the decision boundary."""
    global classifier, criterion, opt, x, y, iteration_count

    if classifier is None or criterion is None or opt is None:
        print("Model not initialized. Please call initialize_model first.")
        return to_js([]) # Return empty boundary if not ready

    # --- Forward pass ---
    y_pred_proba = classifier.forward(x) # Get predicted probabilities
    loss = criterion.forward(y_pred_proba, y) # Calculate loss

    # --- Backward pass ---
    dLdy = criterion.backward() # Get gradient of loss w.r.t predicted probabilities
    classifier.backward(dLdy, opt) # Backpropagate gradients and update weights

    # --- Calculate Accuracy ---
    y_pred_labels = (y_pred_proba.reshape(-1) > 0.5).astype(int) # Convert probabilities to 0/1 labels
    accuracy = np.mean(y_pred_labels == y) # Compare with true labels

    # --- Calculate Decision Boundary for Visualization ---
    grid_size = 72 # Density of the grid for visualization
    x1_grid = np.linspace(0, 1, grid_size)
    x2_grid = np.linspace(0, 1, grid_size)
    X1_mesh, X2_mesh = np.meshgrid(x1_grid, x2_grid) # Create meshgrid
    # Flatten meshgrid and create input pairs for the classifier
    X_grid = np.vstack([X1_mesh.ravel(), X2_mesh.ravel()]).T
    # Get probability predictions for the grid points
    y_grid_proba = classifier.forward(X_grid).reshape(-1)

    # Define boundary as points where probability is between 0.3 and 0.7
    boundary_indices = (y_grid_proba > 0.3) & (y_grid_proba < 0.7)
    boundary_points = X_grid[boundary_indices]

    # --- Logging and Iteration Count ---
    iteration_count += 1
    if iteration_count % 50 == 0: # Log every 50 iterations
        print(f"Iter: {iteration_count}, Loss: {loss:.4f}, Acc: {accuracy:.4f}")

    # --- Handle potential NaN loss ---
    if np.isnan(loss) or np.isinf(loss):
        print(f"Warning: Loss is {loss} at iteration {iteration_count}. Re-initializing model.")
        initialize_model() # Reset if training becomes unstable
        return to_js([]) # Return empty boundary after reset

    # Convert boundary points to list and then to JS object
    return to_js(boundary_points.tolist())