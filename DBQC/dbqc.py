# -----------------------------------------------------------------------------
# Distributed under the GNU General Public License.
#
# Contributors: Nicolas Melo (nmo@cin.ufpe.br)
#               Rodrigo Sousa (--)
#               Adenilton Silva ajsilva@cin.ufpe.br
# -----------------------------------------------------------------------------
# References:
#
# Schuld, M., Fingerhuth, M., & Petruccione, F. (2017). Implementing a distance
# -based classifier with a quantum interference circuit. EPL (Europhysics 
# Letters), 119(6), 60002.
# -----------------------------------------------------------------------------
# File description:
#
# Distance-based quantum classifier implementation on qiskit
# -----------------------------------------------------------------------------
# Notes:
#
# These codes are a replication and were made from
# https://github.com/markf94/ibmq_code_epl_119_60002.
# This file contains slight modifications to eliminate the hard-code present in 
# the original implementation
# -----------------------------------------------------------------------------

from qiskit import (QuantumCircuit, ClassicalRegister, QuantumRegister,
                    execute, BasicAer, IBMQ)
from qiskit.tools.monitor import job_monitor

class DBQC:
    
    def initializeRegisters(self):
        """
        Creates quantum and classical registers
        """
        
        self.qr = QuantumRegister(4)
        self.cr = ClassicalRegister(4)
        
        # Name the individual qubits in quantum register
        self.ancilla_qubit = self.qr[0]
        self.index_qubit = self.qr[1]
        self.data_qubit = self.qr[2]
        self.class_qubit = self.qr[3]
        
    def createCircuit(self, test, training):
        """
        Creates the quantum circuit loading vector angles 'test' and 'training'
        """
        
        # Create empty quantum circuit
        self.qc = QuantumCircuit(self.qr, self.cr)
        
        # Put the ancilla and the index qubits into uniform superposition
        self.qc.h(self.ancilla_qubit)
        self.qc.h(self.index_qubit)
        
        # Load the vector to be classified
        self.load_test_vector(test)
        
        # Flip the ancilla qubit moves the input vector to the |0> 
        # state of the ancilla
        self.qc.x(self.ancilla_qubit)
        self.qc.barrier()
        
        # Load the first training vector
        self.load_training_vector(training[0])
        
        # Flip the index qubit moves the first training vector to the |0> 
        # state of the index qubit
        self.qc.x(self.index_qubit)
        self.qc.barrier()
        
        # Load the second training vector
        self.load_training_vector(training[1])
        
        # The class qubit is flipped conditioned on the index qubit being |1>
        self.qc.cx(self.index_qubit, self.class_qubit)
        self.qc.barrier()
        
        # The Hadamard gate interferes the copies of the test vector with the
        # training vectors
        self.qc.h(self.ancilla_qubit)
        self.qc.barrier()
        
        # Measure the ancilla qubit and store the result in the classical
        # register
        self.qc.measure(self.qr, self.cr)
        
        return self.qc
    
    def load_test_vector(self, theta):
        """
        Loads the test vector from its theta angle
        """
        
        self.qc.cx(self.ancilla_qubit, self.data_qubit)
        self.qc.ry(theta*-1, self.data_qubit) 
        self.qc.cx(self.ancilla_qubit, self.data_qubit) 
        self.qc.ry(theta, self.data_qubit)
        
        self.qc.barrier()
        
    def load_training_vector(self, theta):
        """
        Loads the training vector from its theta angle
        """
        
        self.qc.ccx(self.ancilla_qubit, self.index_qubit, self.data_qubit)
        self.qc.cx(self.index_qubit, self.data_qubit)
        self.qc.ry(theta, self.data_qubit)
        self.qc.cx(self.index_qubit, self.data_qubit)
        self.qc.ry(theta*-1, self.data_qubit)
        
        self.qc.ccx(self.ancilla_qubit, self.index_qubit, self.data_qubit)
        self.qc.cx(self.index_qubit, self.data_qubit)
        self.qc.ry(theta*-1, self.data_qubit)
        self.qc.cx(self.index_qubit, self.data_qubit)
        self.qc.ry(theta, self.data_qubit)
        
        self.qc.barrier()
    
    def runSim(self, quantum_circuit):
        """
        Compiles and runs the quantum circuit on a simulator backend
        """
        
        backend_sim = BasicAer.get_backend('qasm_simulator')
        job_sim = execute(quantum_circuit, backend_sim, shots=1024)

        return job_sim.result()
    
    def runReal(self, quantum_circuit):
        """
        Compiles and runs the quantum circuit on a real quantum processor
        """
        
        # Update/save/load IBM account credentials
        #IBMQ.update_account()
        #IBMQ.save_account('MY_API_TOKEN')
        IBMQ.load_account()
        
        provider = IBMQ.get_provider(group='open')
        backend_real = provider.get_backend('ibmq_essex')
        job = execute(quantum_circuit, backend_real, shots=1024)
        
        # Monitor the status of a IBMQ job instance
        job_monitor(job)
        
        return job.result()
    
    def interpret_results(self, result_counts):
        """
        Interprets the results of the measurement of the ancilla qubit.
        Then computing the statistics of the class qubit
        """
        
        total_samples = sum(result_counts.values())
        
        # Define lambda function that retrieves only results where the ancilla 
        # is in the |0> state
        post_select = lambda counts: [(state, occurences) for state, occurences 
                                      in counts.items() if state[-1] == '0']

        # Perform the postselection by applying post_select lambda function to
        # the results observed
        postselection = dict(post_select(result_counts))
        postselected_samples = sum(postselection.values())

        # Probability of success of the post-selection
        ps_prob = postselected_samples/total_samples

        # Define lambda function that retrieves results of the post-selection
        # where the result is equal to the indicated binary class
        retrieve_class = lambda binary_class: [occurences for state, occurences 
                                               in postselection.items() if 
                                               state[0] == str(binary_class)]

        # Resulting probability for each class
        prob_c0 = sum(retrieve_class(0))/postselected_samples
        prob_c1 = sum(retrieve_class(1))/postselected_samples

        return prob_c0, prob_c1, ps_prob
    
    def classifySim(self, test_vector, training_set):
        """
        Classifies the test vector with the distance-based quantum classifier 
        using simulator
        """
        
        self.initializeRegisters()
        qc = self.createCircuit(test_vector, training_set)

        result = self.runSim(qc)

        prob_c0, prob_c1, ps_prob = self.interpret_results(result.get_counts(qc))

        # The class most likely to occur classifies the input vector
        if prob_c0 > prob_c1:
            return prob_c0, prob_c1, ps_prob, 0
        elif prob_c0 < prob_c1:
            return prob_c0, prob_c1, ps_prob, 1
        else:
            return 'inconclusive. 50/50 results'
        
    def classifyReal(self, test_vector, training_set):
        """
        Classifies the test vector with the distance-based quantum classifier 
        using a real quantum processor
        """
        
        self.initializeRegisters()
        qc = self.createCircuit(test_vector, training_set)

        realExec = self.runReal(qc)
        
        prob_c0, prob_c1, ps_prob = self.interpret_results(realExec.get_counts(qc))

        # The class most likely to occur classifies the input vector
        if prob_c0 > prob_c1:
            return prob_c0, prob_c1, ps_prob, 0
        elif prob_c0 < prob_c1:
            return prob_c0, prob_c1, ps_prob, 1
        else:
            return 'Inconclusive. 50/50 results'