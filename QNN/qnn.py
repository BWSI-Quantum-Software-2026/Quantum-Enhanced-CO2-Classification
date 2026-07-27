import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp
from data.standardization import load_and_prepare

X, Y = load_and_prepare()
num_qubits = X.shape[1]

input_params = ParameterVector("x", num_qubits)
weight_params = []

qc = QuantumCircuit(num_qubits)

# Superposition Layer
qc.h(0)
qc.h(1)
qc.h(2)

# Feature Encoding
for i in range(num_qubits):
    qc.ry(input_params[i], i)

# Variational Block
depth = 3
entanglement_type = 2

for layer in range(depth):
    # Entanglement Types 
    if entanglement_type == 1:
        qc.cx(0, 1)
        qc.cx(1, 2)
    else:
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 0)

    # Rotations
    theta = [Parameter(f"theta_{layer}_{i}") for i in range(num_qubits)]
    phi = [Parameter(f"phi_{layer}_{i}") for i in range(num_qubits)]
    omega = [Parameter(f"omega_{layer}_{i}") for i in range(num_qubits)]

    weight_params.extend(theta)
    weight_params.extend(phi)
    weight_params.extend(omega)

    qc.rz(theta[0], 0); qc.ry(phi[0], 0); qc.rz(omega[0], 0)
    qc.rz(theta[1], 1); qc.ry(phi[1], 1); qc.rz(omega[1], 1)
    qc.rz(theta[2], 2); qc.ry(phi[2], 2); qc.rz(omega[2], 2)

# Observable
observable = SparsePauliOp.from_list([("ZII", 1.0)])
estimator = Estimator()

def forward(x, w):
    bind_dict = {}

    # Bind Input
    for i in range(num_qubits):
        bind_dict[input_params[i]] = x[i]

    # Bind Weights
    for param, value in zip(weight_params, w):
        bind_dict[param] = value

    bound_qc = qc.bind_parameters(bind_dict)
    result = estimator.run(bound_qc, observable).result()
    return result.values[0]