# -----------------------------------------------------------------------------
# Distributed under the GNU General Public License.
#
# Contributors: Nicolas Melo (nmo@cin.ufpe.br)
#               Rodrigo Sousa (--)
#               Adenilton Silva ajsilva@cin.ufpe.br
# -----------------------------------------------------------------------------
# References:
#
# ESANN2020 paper
#
# Schuld, M., Fingerhuth, M., & Petruccione, F. (2017). Implementing a distance
# -based classifier with a quantum interference circuit. EPL (Europhysics 
# Letters), 119(6), 60002.
# -----------------------------------------------------------------------------
# File description:
#
# Quantum one-class classifier implementation on qiskit
# -----------------------------------------------------------------------------
# Notes:
#
# This classifier is based on the classifier presented in 
# https://arxiv.org/abs/1703.10793
# -----------------------------------------------------------------------------

import qoc
import util
import statistics
from sklearn.metrics import mean_squared_error

def expQOC(classe, backend):
    """
    Performs the quantum one-class classifier for each test vector of one 
    dataset class (0, 1, 2). The execution can be either in a simulator ('sim') or in a 
    real quantum processor ('real')
    """
    
    # Define classes and features of the dataset
    classes = [0,1]
    features = [0,1]
    
    # Load iris dataset
    X, y = util.load_iris(classes=classes, features=features)
    
    # Apply preprocess to iris dataset
    X = util.preprocess(X)
    
    # Get the centroid of the specified class
    if(classe==0):
        cent = util.centroid(X[0:50])
    elif(classe==1):
        cent = util.centroid(X[50:100])
        
    # Take the 2 training vectors closest to the centroid of the class
    training_set = util.minDistances(X, cent, 2)
    
    # Get the training angle for each vector in training set
    training_angles = []
    for i in range(len(training_set)):
        training_angles.append(util.get_theta(X[training_set[i]])/4)
        
    # Initiate an instance of the quantum one-class classifier
    classifier = qoc.QOC()
    
    # Counter to the success of the classification
    success = 0
    
    # Store predicted results
    predict = []
    
    # Run QOC for each test vector that is not in the training set
    for i in range(100):
        print(i)
        if i not in training_set:
            # Get the test angle to classify
            classify = util.get_theta(X[i])/2
            
            # Run the classifier in a simulator or in a real quantum processor
            if(backend == 'sim'):
                total_samples, ancilla_counts, ancilla_samples, ancilla_prob = classifier.classifySim(
                        test_vector=classify, training_set=training_angles)
            elif(backend == 'real'):
                total_samples, ancilla_counts, ancilla_samples, ancilla_prob = classifier.classifyReal(
                        test_vector=classify, training_set=training_angles)
            else:
                return 'Invalid backend'
            
            # Increase the success counters of the classification.
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
            
            # Store the predicted results
            predict.append(ancilla_prob)
            
    # Create list of true values
    if (classe == 0):
        true = [1]*48 + [0]*50
    if (classe == 1):
        true = [0]*50 + [1]*48
    
    # Calculate statistics
    mse = mean_squared_error(true, predict)
    stdev = statistics.stdev(predict)
    variance = statistics.variance(predict)
    
    # Prints
    print(f'Success probability: {success/98}')
    print(f'Fail probability: {1-(success/98)}\n')    
    print(f'Mean square error: {mse}\n')    
    print(f'Standard deviation: {stdev}\n')    
    print(f'Variance: {variance}\n')    
    print(f'Predict: {predict}')