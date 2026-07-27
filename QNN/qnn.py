import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp
from data.standardization import load_and_prepare

# standardization
X, Y = load_and_prepare()

# qubit allocation
num_qubits = X.shape[1]

input_params = ParameterVector("x", num_qubits) # Holding data rotations, Feature Vector
weight_params = [] # Stores trainable parameters

qc = QuantumCircuit(num_qubits) # 3 qubit quantum circuit

# Superposition Layer
qc.h(0)
qc.h(1)
qc.h(2)

# Data encoding layer
for i in range(num_qubits):
    qc.ry(input_params[i], i)

# Variational Block
depth = 3 # Number of times to repeat variational layer
entanglement_type = 2 # Choose from EntanglingLayer and StronglyEntanglingLayer in the paper

for layer in range(depth): # Repeats for depth
    # Entanglement Types 
    if entanglement_type == 1: # EntanglingLayer
        qc.cx(0, 1)
        qc.cx(1, 2)
    else: # StronglyEntanglingLayer
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 0)

    # Rotations
    theta = [Parameter(f"theta_{layer}_{i}") for i in range(num_qubits)] # First angle on qubit i
    phi = [Parameter(f"phi_{layer}_{i}") for i in range(num_qubits)] # Second angle on qubit i
    omega = [Parameter(f"omega_{layer}_{i}") for i in range(num_qubits)] # Third angle on qubit i

    # Append angle parameters to weight_params
    weight_params.extend(theta)
    weight_params.extend(phi)
    weight_params.extend(omega)

    qc.rz(theta[0], 0); qc.ry(phi[0], 0); qc.rz(omega[0], 0) # Able to go to any point on bloch sphere, qubit 1
    qc.rz(theta[1], 1); qc.ry(phi[1], 1); qc.rz(omega[1], 1) # Qubit 2 rotations
    qc.rz(theta[2], 2); qc.ry(phi[2], 2); qc.rz(omega[2], 2) # Qubit 3 rotations

# Observable
observable = SparsePauliOp.from_list([("ZII", 1.0)]) # Measure outputted value of qubit 0
estimator = Estimator()

def forward(x, w): # first 3 inputs and weight values
    bind_dict = {} # Maps parameters to numerical values

    # Bind Input
    for i in range(num_qubits):
        bind_dict[input_params[i]] = x[i] # Rotations into data encoding layer

    # Bind Weights
    for param, value in zip(weight_params, w):
        bind_dict[param] = value # Weights into variational layers

    bound_qc = qc.bind_parameters(bind_dict) # Parameters with real numbers
    result = estimator.run(bound_qc, observable).result() # Runs estimator on circuit and observable
    return result.values[0] # Returns scalar output of QNN for that sample

# Training loss with MAE, error on training data
def loss_train(w):
    preds = np.array([forward(X_train[i], w) for i in range(len(X_train))]) # predictions
    return np.mean(np.abs(preds - Y_train))

# Validation loss with MAE, generalization to nontrained data
def loss_val(w):
    preds = np.array([forward(X_val[i], w) for i in range(len(X_val))]) # predictions
    return np.mean(np.abs(preds - Y_val))

# Gradient estimation
def grad_batch(w, Xb, Yb, epsilon = 0.001): # batch of data
    g = np.zeros_like(w) # empty vector
    for k in range(len(w)):
        w_plus = w.copy(); w_plus[k] += epsilon # shifts up slightly
        w_minus = w.copy(); w_minus[k] -= epsilon # shifts down slightly
        preds_plus = np.array([forward(Xb[i], w_plus) for i in range(len(Xb))]) # prediction with slight upward shift
        preds_minus = np.array([forward(Xb[i], w_minus) for i in range(len(Xb))]) # prediction with slight downward shift
        g[k] = (np.mean(np.abs(preds_plus - Yb)) - np.mean(np.abs(preds_minus - Yb))) / (2 * epsilon) # gradient of loss to weight
    return g # tells how to update weights

# Training setup (from paper)
learning_rate = 0.1 # how large each update to weight is
batch_size = 10 # number of data samples per batch (days)
validation_split = 0.1 # what amount of data is used for validation
epochs = 30 # number of times repeating training, updating epoch

split_index = int(len(X) * (1 - validation_split)) # math to determine which of the dataset is used for training and which is used for validation (first 90% for training and last 10% for validation)
X_train, Y_train = X[:split_index], Y[:split_index] # training inputs and targets
X_val, Y_val = X[split_index:], Y[split_index:] # validation inputs and targets

w = np.random.uniform(-0.1, 0.1, size = len(weight_params)) # initial guess for rotations

def get_batch():
    index = np.random.choice(len(X_train), batch_size, replace = False) # randomly chooses batch and cannot select same batch
    return X_train[index], Y_train[index] 

for epoch in range(epochs):
    Xb, Yb = get_batch() # gets random training data
    g = grad_batch(w, Xb, Yb) # finds gradient of batch loss to weight
    w -= learning_rate * g # updates weights, if g is negative it will increase and if g is positive it will decrease

    print(f"Epoch {epoch}, train loss = {loss_train(w):.6f}, validation loss = {loss_val(w):.6f}") # finds training and validation loss and prints both to 6 decimals