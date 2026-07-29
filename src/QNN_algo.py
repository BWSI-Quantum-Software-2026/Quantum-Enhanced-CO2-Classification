import numpy as np
import sys, os
# project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(project_root)
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.primitives import StatevectorEstimator
from qiskit.primitives.containers import ObservablesArray
from qiskit.quantum_info import SparsePauliOp


class weatherQNN:
    
    def __init__(
        self, 
        num_qubits, 
        depth = 3, 
        entanglement_type = 2
    ):
        self.num_qubits = num_qubits
        self.depth = depth

        self.input_params = ParameterVector('x', num_qubits)
        self.weight_params = []

        self.qc = QuantumCircuit(num_qubits)

        self.build_circuit(entanglement_type)

        self.observable = SparsePauliOp.from_list(
            [('Z'+'I'*(num_qubits-1), 1.0)]
        )

        self.estimator = StatevectorEstimator()



    def build_circuit(self, entanglement_type):
        #Superposition 
        for i in range(self.num_qubits):
            self.qc.h(i)

        for i in range(self.num_qubits):
            self.qc.ry(self.input_params[i], i)

        for layer in range(self.depth):
            # the procedure both work for type 1 and type2
            for i in range(self.num_qubits-1):
                self.qc.cx(i, i+1)

            if entanglement_type == 2:
                if self.num_qubits > 2:
                    self.qc.cx(self.num_qubits-1, 0)

            for i in range(self.num_qubits):
                theta = Parameter(f'theta_{layer}_{i}')
                phi = Parameter(f'phi_{layer}_{i}')
                omega = Parameter(f'omega_{layer}_{i}')

                self.weight_params.extend(
                    [theta, phi, omega]
                )

                self.qc.rz(theta, i)
                self.qc.ry(phi, i)
                self.qc.rz(omega, i)

    # # Observable
    # observable = SparsePauliOp.from_list([("ZII", 1.0)]) # Measure outputted value of qubit 0
    # observables = ObservablesArray([observable])
    # estimator = StatevectorEstimator()


    def forward(self, x, w): # first 3 inputs and weight values
        bind_dict = {} # Maps parameters to numerical values

        estimator = StatevectorEstimator()

        # Bind Input
        for i, param in enumerate(self.input_params):
            bind_dict[param] = x[i] # Rotations into data encoding layer

        # Bind Weights
        for param, value in zip(self.weight_params, w):
            bind_dict[param] = value # Weights into variational layers

        bound_qc = self.qc.assign_parameters(bind_dict) # Bind parameters with real numbers

        result = self.estimator.run(
            [(bound_qc, self.observable)]
        ).result()
        
        return float(result[0].data.evs) # Returns scalar output of QNN for that sample


    # Gradient estimation
    #Training loss with MAE, error on training data
    def batch_loss(self, w, Xb, Yb):

        predictions = []

        for x in Xb:
            pred = self.forward(x, w)
            predictions.append(pred)

        loss = np.mean(
            np.abs(np.array(predictions)-Yb)
        )

        return loss

    #Gradient estimation
    def grad_batch(self, w, Xb, Yb):

        g = np.zeros_like(w) # empty vector

        for k in range(len(w)):
            w_plus = w.copy(); 
            w_minus = w.copy(); 

            w_plus[k] += np.pi/2 #shift up
            w_minus[k] -= np.pi/2 #shift down

            loss_plus = self.batch_loss(w_plus, Xb, Yb) # prediction with slight upward shift
            loss_minus = self.batch_loss(w_minus, Xb, Yb) # prediction with slight downward shift

            g[k] = (loss_plus-loss_minus)/2 # gradient of loss to weight

        return g # tells how to update weights
