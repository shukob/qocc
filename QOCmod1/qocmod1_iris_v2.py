"""
Classificador que recebe vetores com 4 features
"""

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, execute, BasicAer
import numpy as np
import util


def initQC(n):
    q1 = QuantumRegister(n/2)
    q2 = QuantumRegister(n/2)
    c = ClassicalRegister(n)
    qc = QuantumCircuit(q1, q2, c)
    
    return q1, q2, c, qc


def simQasm(qcircuit):
    backend_sim = BasicAer.get_backend('qasm_simulator')
    job_sim = execute(qcircuit, backend_sim, shots=1000)
    result = job_sim.result()
    result_counts = result.get_counts(qcircuit)
    
    return result_counts


def classifySim(test_vector, training_set):
    
    qc = circuit(test_vector, training_set)
    
    result_counts = simQasm(qc)
    total_samples = sum(result_counts.values())
    
    # Define lambda function that retrieves only results where the ancilla is in the |0> state
    ancilla_func = lambda counts: [(state, occurences) for state, occurences in counts.items() if state[-1] == '0']
    
    # Retrive counting values for the ancilla qubit by applying ancilla_func
    ancilla_counts = dict(ancilla_func(result_counts))
    ancilla_samples = sum(ancilla_counts.values())
    
    # Probability of ancilla qubit is in the |0> state
    ancilla_prob = ancilla_samples/total_samples
    
    return total_samples, ancilla_counts, ancilla_samples, ancilla_prob


def experiment(classe):
    
    # classes and features of the iris dataset
    classes = [0,1]
    features = [0,1,2,3]
    
    X, y = util.load_iris(classes=classes, features=features)
    X = util.preprocess(X)
    
    if(classe==0):
        cent = util.centroid2(X[0:50])
        print(cent)
        training_set = util.minDistances(X[0:50], cent, 2)
    elif(classe==1):
        cent = util.centroid2(X[50:100])
        training_set = util.minDistances(X[50:100], cent, 2)
        training_set = np.array(training_set) + 50
    
    training_vectors = []
    
    for i in range(len(training_set)):
        training_vectors.append(X[training_set[i]])
    
    success = 0
    predict = []
    
    for i in range(100):
        print(i)
        
        if i not in training_set:
            classify = X[i]
            
            total_samples, ancilla_counts, ancilla_samples, ancilla_prob = classifySim(
                        test_vector=classify, training_set=training_vectors)
            
            if(classe==0):
                if (i < 50):
                    if (ancilla_prob > 0.5):
                        success += 1
                else:
                    if (ancilla_prob < 0.5):
                        success += 1
                    
            if(classe==1):
                if ((i < 50) and (ancilla_prob < 0.5)):
                    success += 1
                elif((i >= 50) and (ancilla_prob > 0.5)):
                    success += 1
            
            predict.append(ancilla_prob)
            
    print(f'Success probability: {success/98}')
    print(f'Fail probability: {1-(success/98)}\n')
    print(f'Predict: {predict}')
    
    
def amplitude_encoding(input_vector, qcirc, qRegAux, qRegAmp, typeV):
    """
    load real vector x to the amplitude of a quantum state
    """
    #num_qubits = np.log2(len(input_vector))
    quantumAux = qRegAux
    quantum_input = qRegAmp
    qcircuit = qcirc
    newx = np.copy(input_vector)
    betas = []
    _recursive_compute_beta(newx, betas)
    return _generate_circuit(betas, qcircuit, quantumAux, quantum_input, typeV)

def _recursive_compute_beta(input_vector, betas):
    if len(input_vector) > 1:
        new_x = []
        beta = []
        for k in range(0, len(input_vector), 2):
            norm = np.sqrt(input_vector[k] ** 2 + input_vector[k + 1] ** 2)
            new_x.append(norm)
            if norm == 0:
                beta.append(0)
            else:
                if input_vector[k] < 0:
                    beta.append(2 * np.pi - 2 * np.arcsin(input_vector[k + 1] / norm)) ## testing
                else:
                    beta.append(2 * np.arcsin(input_vector[k + 1] / norm))
        _recursive_compute_beta(new_x, betas)
        betas.append(beta)
            
            
def _generate_circuit(betas, qcircuit, quantumAux, quantum_input, typeV):
    
    if (typeV=='test'):
        numberof_controls = 0  # number of controls
        control_bits = []
        for angles in betas:
            if numberof_controls == 0:
                #qcircuit.ry(angles[0], quantum_input[0])
                qcircuit.cry(angles[0], quantumAux[0], quantum_input[0])
                numberof_controls += 1
                control_bits.append(quantum_input[0])
            else:
                for k, angle in enumerate(reversed(angles)):
                    _index(k, qcircuit, control_bits, numberof_controls)
                        
                    qcircuit.mcry(angle,
                                  [quantumAux[0]] + control_bits,
                                  quantum_input[numberof_controls],
                                  None,
                                  mode='noancilla')
                        
                    _index(k, qcircuit, control_bits, numberof_controls)
                control_bits.append(quantum_input[numberof_controls])
                numberof_controls += 1
        return qcircuit
    
    else:
        numberof_controls = 0  # number of controls
        control_bits = []
        for angles in betas:
            if numberof_controls == 0:
                qcircuit.mcry(angles[0], [quantumAux[0], quantumAux[1]], quantum_input[0], None, mode='noancilla')
                numberof_controls += 1
                control_bits.append(quantum_input[0])
            else:
                for k, angle in enumerate(reversed(angles)):
                    _index(k, qcircuit, control_bits, numberof_controls)
                        
                    qcircuit.mcry(angle,
                                  [quantumAux[0], quantumAux[1]] + control_bits,
                                  quantum_input[numberof_controls],
                                  None,
                                  mode='noancilla')
                        
                    _index(k, qcircuit, control_bits, numberof_controls)
                control_bits.append(quantum_input[numberof_controls])
                numberof_controls += 1
        return qcircuit


def _index(k, circuit, control_qubits, numberof_controls):
    binary_index = '{:0{}b}'.format(k, numberof_controls)
    #for j, qbit in enumerate(reversed(control_qubits)):
    for j, qbit in enumerate(control_qubits):
        if binary_index[j] == '1':
            circuit.x(qbit)
    
    
def circuit(test, training):
    
    q1, q2, c, qc = initQC(4)
    
    ancilla = q1[0]
    index = q1[1]
    #data_1 = q2[0]
    #data_2 = q2[1]
    
    # ancilla and index in uniform superposition
    qc.h(ancilla)
    qc.h(index)
    qc.barrier()
    
    # load test vector
    #loadTest(test, qc, ancilla, data_1, data_2)
    amplitude_encoding(test, qc, q1, q2, 'test')
    
    # flip the ancilla qubit moves the input vector to the |0> state of the ancilla
    qc.x(ancilla)
    qc.barrier()
    
    # load first training vector
    #loadTraining(training[0], qc, ancilla, index, data_1, data_2)
    amplitude_encoding(training[0], qc, q1, q2, 'training')
    
    # flip the index qubit moves the first training vector to the |0> state of the index qubit
    qc.x(index)
    qc.barrier()
    
    # load second training vector
    #loadTraining(training[1], qc, ancilla, index, data_1, data_2)
    amplitude_encoding(training[1], qc, q1, q2, 'training')
    
    # the Hadamard gate interferes the copies of the test vector with the training vectors
    qc.h(ancilla)
    
    # measure ancilla qubit
    qc.measure(ancilla, c[0])
    
    return qc