import qiskit.circuit.random
from cuquantum import contract, CircuitToEinsum

qc = qiskit.circuit.random.random_circuit(num_qubits=8, depth=7)
converter = CircuitToEinsum(qc, backend='cupy')
print(converter)