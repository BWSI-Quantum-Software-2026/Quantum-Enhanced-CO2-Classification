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
        self.weight_params = []#not including a, b

        self.qc = QuantumCircuit(num_qubits)

        self.build_circuit(entanglement_type)

        # paulis = []
        # for i in range(num_qubits):
        #     s = ["I"] * num_qubits
        #     s[i] = "Z"
        #     paulis.append(("".join(s), 1.0/num_qubits))

        # self.observable = SparsePauliOp.from_list(paulis)
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

    # observables = ObservablesArray([observable])
    
    @property
    def n_circuit_params(self):
        return len(self.weight_params)

    @property
    def n_total_params(self):
        return self.n_circuit_params+2

    def split_w(self, w):
        w_circuit = w[:self.n_circuit_params]
        a = w[-2]
        b = w[-1]
        return w_circuit, a, b

    def forward_raw(self, x, w_circuit):
        bind_dict = {}#maps parameter to numerical values

        estimator = StatevectorEstimator()

        #Bind Input
        for i, param in enumerate(self.input_params):
            bind_dict[param] = x[i] #Rotations into data encoding layer

        #Bind Weights
        for param, value in zip(self.weight_params, w_circuit):
            bind_dict[param] = value #Weights into variational layers

        bound_qc = self.qc.assign_parameters(bind_dict)

        result = self.estimator.run(
            [(bound_qc, self.observable)]
        ).result()

        return float(result[0].data.evs)

    
    def forward(self, x, w): # first inputs and weight values
        w_circuit, a, b = self.split_w(w)
        raw = self.forward_raw(x, w_circuit)
        return a*raw + b


    #Training loss with MSE, error on training data
    def batch_loss(self, w, Xb, Yb):

        predictions = []

        for x in Xb:
            pred = self.forward(x, w)
            predictions.append(pred)

        loss = np.mean(
            np.abs(np.array(predictions)-Yb)**2
        )

        return loss

    #return the. prediction for the whole batch
    def forward_batch_raw(self, w, Xb):
        #raw gradients for weights, not including a, b
        return np.array([self.forward(x, w) for x in Xb])

    def forward_batch(self, w, Xb):
        w_circuit, a, b = self.split_w(w)
        raw = self.forward_batch_raw(w, Xb)
        return a*raw + b

    def batch_loss(self, w, Xb, Yb):
        predictions = self.forward_batch_raw(w, Xb)
        loss = np.mean(np.abs(predictions-Yb)**2)
        return loss


    #Gradient estimation
    def grad_batch(self, w, Xb, Yb):

        N = len(Xb)
        w_circuit, a, b = self.split_w(w)
        g = np.zeros_like(w) # empty vector

        # for k in range(len(w)):
        #     w_plus = w.copy(); 
        #     w_minus = w.copy(); 

        #     w_plus[k] += np.pi/2 #shift up
        #     w_minus[k] -= np.pi/2 #shift down

        #     loss_plus = self.batch_loss(w_plus, Xb, Yb) # prediction with slight upward shift
        #     loss_minus = self.batch_loss(w_minus, Xb, Yb) # prediction with slight downward shift

        #     g[k] = (loss_plus-loss_minus)/2 # gradient of loss to weight

        raw = self.forward_batch_raw(w_circuit, Xb) #the raw prediction
        f = a*raw+b#the prediction
        dL_df = 2*(f-Yb)/N #the derivative for MSE between Loss and prediction

        for k in range(len(w)):
            w_plus = w.copy()
            w_minus = w.copy()
            w_plus[k] += np.pi/2
            w_minus[k] -= np.pi/2
    
            raw_plus = self.forward_batch_raw(w_plus, Xb)
            raw_minus = self.forward_batch_raw(w_minus, Xb)

            df_dtheta_k = a*(raw_plus-raw_minus)/2 # f=a*raw+b => df/dtheta_k = a*d(raw)/dtheta_k
            g[k] = np.sum(dL_df*df_dtheta_k)

        g[-2] = np.sum(dL_df*raw) #dL_da
        g[-1] = np.sum(dL_df*1.0) #dL_db
        
        return g # tells how to update weights
