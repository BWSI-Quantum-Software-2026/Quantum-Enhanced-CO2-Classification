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