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
# Distance-based quantum classifier experiment implementation on qiskit
# -----------------------------------------------------------------------------
# Notes:
#
# These codes are a replication and were made from
# https://github.com/markf94/ibmq_code_epl_119_60002.
# This file contains slight modifications to eliminate the hard-code present in 
# the original implementation
# -----------------------------------------------------------------------------

import dbqc
import util
import statistics
from sklearn.metrics import mean_squared_error

def expDBQC(backend):
    """
    Performs the distance-based quantum classifier for each test vector in the 
    dataset. The execution can be either in a simulator ('sim') or in a real 
    quantum processor ('real')
    """
    
    # Define classes and features of the dataset
    classes = [0,1]
    features = [0,1]
    
    # Load iris dataset
    X, y = util.load_iris(classes=classes, features=features)
    
    # Apply preprocess to iris dataset
    X = util.preprocess(X)
    
    # Get the centroid of each class
    # 0-49 -> class 1
    # 50-99 -> class 2
    # 100-149 -> class 3
    cent0 = util.centroid(X[0:50])
    cent1 = util.centroid(X[50:100])
    
    # Take the training vector closer to the centroid of each class
    training_set = []
    training_set.append(util.minDistances(X, cent0, 1)[0])
    training_set.append(util.minDistances(X, cent1, 1)[0])
    
    # Get the training angle for each vector in training set
    training_angles = []
    for i in range(len(training_set)):
        training_angles.append(util.get_theta(X[training_set[i]])/4)
    
    # Initiate an instance of the distance-based quantum classifier
    classifier = dbqc.DBQC()
    
    # Counter to the success of the post-selection and classification
    ps_success = 0
    success = 0
    
    # Store predicted results for when post-selection occurs and when not
    predict_ps = []
    predict_no_ps = []
    
    # Store post-selection probabilities
    ps_probs =[]
    
    # Run DBQC for each test vector that is not in the training set
    for i in range(100):
        print(i)
        if i not in training_set:
            # Get the test angle to classify
            classify = util.get_theta(X[i])/2
            
            # Run the classifier in a simulator or in a real quantum processor
            if(backend == 'sim'):
                prob_c0, prob_c1, ps_prob, class_result = classifier.classifySim(
                        test_vector=classify, training_set=training_angles)
            elif(backend == 'real'):
                prob_c0, prob_c1, ps_prob, class_result = classifier.classifyReal(
                        test_vector=classify, training_set=training_angles)
            else:
                return 'Invalid backend'
            
            # Increase the success counters of post-selection and classification.
            # Store the predicted results in the appropriate lists
            if (ps_prob > 0.5):
                ps_success += 1
                if (i < 50):
                    predict_ps.append(prob_c0)
                    predict_no_ps.append(prob_c0)
                    if (class_result == 0):
                        success += 1
                else:
                    predict_ps.append(prob_c1)
                    predict_no_ps.append(prob_c1)
                    if (class_result == 1):
                        success += 1
            else:
                predict_no_ps.append(0)
                
            ps_probs.append(ps_prob)
    
    # Create list of true values
    true_ps = [1]*ps_success
    true_no_ps = [1]*98
    
    # Calculate statistics
    mse_ps = mean_squared_error(true_ps, predict_ps)
    mse_no_ps = mean_squared_error(true_no_ps, predict_no_ps)
    
    stdev_ps = statistics.stdev(predict_ps)
    stdev_no_ps = statistics.stdev(predict_no_ps)
    
    variance_ps = statistics.variance(predict_ps)
    variance_no_ps = statistics.variance(predict_no_ps)
    
    # Prints
    print('\n')
    print(f'Overall success classification probability: {success/98}')
    print(f'Overall fail classification probability: {1-(success/98)}\n')
    print(f'Success probability given post-selection: {success/ps_success}')
    print(f'Fail probability given post-selection: {1-(success/ps_success)}\n')
    
    print(f'Post-selection success: {ps_success}\n')
    print(f'Post-selection probabilities: {ps_probs}\n')
    print(f'Post-selection probability success: {ps_success/98}')
    print(f'Post-selection probability fail: {1-(ps_success/98)}\n')
    
    print(f'Mean square error (with post-selection success): {mse_ps}')
    print(f'Mean square error (without post-selection success): {mse_no_ps}\n')
    print(f'Standard deviation (with post-selection success): {stdev_ps}')
    print(f'Standard deviation (without post-selection success): {stdev_no_ps}\n')
    print(f'Variance (with post-selection success): {variance_ps}')
    print(f'Variance (without post-selection success): {variance_no_ps}\n')
    
    print(f'Predict ps: {predict_ps}\n')
    print(f'Predict no-ps: {predict_no_ps}')