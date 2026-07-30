import sys, os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from qiskit import QuantumCircuit
from qiskit import transpile
from qiskit_aer import AerSimulator
from QNN.qnn_temperature import qnn_temperature
from QNN.qnn_humidity import qnn_humidity
from QNN.qnn_wind_speed import qnn_wind_speed
backend = AerSimulator()
qc_temp, w_temp, forward = qnn_temperature()
qc_hum, w_hum = qnn_humidity()
qc_wspd, w_wspd = qnn_wind_speed()

circuits = [qc_temp, qc_hum, qc_wspd]

compiled = [transpile(circ, backend) for circ in circuits]

for i, circ in enumerate(compiled):
    print(f"QNN {i+1}")
    print("Qubits:", circ.num_qubits)
    print("Depth:", circ.depth())
    print("Ops:", circ.count_ops())


