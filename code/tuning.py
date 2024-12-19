from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.feature_extraction import DictVectorizer
import pandas as pd
import seaborn as sn
import sys
import string
from sklearn.metrics import classification_report,confusion_matrix, ConfusionMatrixDisplay, recall_score, f1_score,precision_score,  accuracy_score
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
lemmatizer = WordNetLemmatizer()

"""NOTE: This script is just for hyperparameter tuning, it gives back a file with the best parameters for each model it tuned."""
    
def extract_features_and_labels(trainingfile):
    """
    Extracts features and labels from a training file containing tokenized text data.

    This function processes an input file where each line contains a token, its 
    corresponding part-of-speech (POS) tag, chunk information, and a label. The 
    file is expected to be space-separated, with each line structured as:
    
    <token> <POS> <chunk> <label>
    
    The function reads the file line by line, splits each line into its respective 
    components, stores the features (token, POS, chunk) in a dictionary, and stores 
    the label separately. The dictionaries of features are collected into one list, 
    and the labels are collected into another list, which are then returned as a tuple.

    Arguments:
        trainingfile (str): The path to the input training file containing tokenized 
                            data and labels. The file is expected to have one token-POS-chunk-label 
                            quartet per line.

    Returns:
        tuple: A tuple containing two elements:
            - A list of dictionaries, where each dictionary contains the token, POS tag, and chunk information.
              The structure of each dictionary is as follows:
              {
                  'token': <string>,
                  'pos': <string>,
                  'lemma': <string>,
                  'shape': <string>
              }
            - A list of labels corresponding to each line in the file. Each label is stored as a string.

    """
    data = []
    targets = []
    previous_token =''
    with open(trainingfile, 'r', encoding='utf8') as infile:
        for line in infile:
            components = line.rstrip('\n').split()
            if len(components) > 0:
                token = components[0]
                pos = components[1]
                lemma = lemmatizer.lemmatize(token)
                shape =''
                for char in token:
                    if char == '.':
                        shape = 'Has_period'
                    elif char =='-':
                        shape = 'Has_hyphen'
                if token.isupper():
                    shape = shape + 'Upper'
                elif token.islower():
                    shape = shape + 'Lower'
                elif token.istitle():
                    shape = shape + 'First_letter_capital'
                elif token.isdigit():
                    shape = shape + 'Digit'
                else:
                    shape = 'Other'
               
                feature_dict = {'token':token, 'pos': pos, 'lemma': lemma, 
                                'shape': shape, 'previous token':previous_token}
                
                data.append(feature_dict)
                targets.append(components[-1])
                previous_token = token
    return data, targets
    
    

def create_classifier(train_features, train_targets, modelname = 'logreg'):
    """
    Creates and trains a logistic regression classifier.

    This function takes in a set of training features (token,pos-tag,chunk-tag) and their corresponding 
    target labels (NE-tag), vectorizes the features using `DictVectorizer`, and trains 
    a logistic regression model on the vectorized features.

    Arguments:
        train_features: A list of dictionaries where each dictionary contains 
                                        feature names as keys and their respective values as the features 
                                        for each instance. 
        train_targets: A list of target labels corresponding to the training instances. 
        modelname: A string that defaults to a logregssion classifier, other options are Naive Bayes (NB) or Support Vector Machine (SVM)

    Returns:
        tuple: A tuple containing:
            - `model`: The trained logistic regression classifier.
            - `vec`: The fitted `DictVectorizer` used to vectorize the training features.
    """
    vec = DictVectorizer()
    features_vectorized = vec.fit_transform(train_features)

    
    if modelname == 'logreg':
        model = LogisticRegression(max_iter=2000)
        distributions = dict(C=uniform(loc=0, scale=4),
                         penalty =['l2'])
        classifier = RandomizedSearchCV(model,distributions, n_jobs =-1,random_state = 0)
        search = classifier.fit(features_vectorized, train_targets)
    if modelname =='NB':
        model = MultinomialNB()
        distributions = dict(alpha=uniform(loc=0, scale=4)) #in a multinomial NB only the Alpha can be optimized
        classifier = RandomizedSearchCV(model,distributions, n_jobs =-1,random_state = 0)
        search = classifier.fit(features_vectorized, train_targets)
    if modelname == 'SVM':
        model = LinearSVC(max_iter = 2000)
        distributions = dict(C=uniform(loc=0, scale=4),
                         penalty =['l2','l1'],
                            loss = ['hinge','squared_hinge'])
        classifier = RandomizedSearchCV(model,distributions, n_jobs =-1,random_state = 0)
        search = classifier.fit(features_vectorized, train_targets)
    
  
    model.fit(features_vectorized, train_targets)
    best_param = [modelname,search.best_params_]
    with open('../models/model_parameters/best_parameters'+modelname+'.txt','w') as outfile:
        outfile.write(str(best_param))
    return model, vec
    

    
def main(argv=None):
    """ 
    Main function to handle command-line arguments and perform machine learning classification.

    This function processes input arguments to train a machine learning classifier
    and use it to classify data from an input file, saving the results to an output file.
    It is designed for simple scenarios and uses `sys.argv` for argument parsing. 
    For more robust argument handling, consider using `argparse`.

    Args:
        argv (list[str], optional): A list of command-line arguments. If not provided,
            `sys.argv` is used. Expected structure:
            - argv[1]: Path to the training file (contains features and labels).
            - argv[2]: Path to the input file (data to classify).
            - argv[3]: Path to the output file (to save classification results).
    """

    #a very basic way for picking up commandline arguments
    if argv is None:
        argv = sys.argv
        
    
    argv =['classification_script','../data/conll2003/conll2003.train.conll','../data/conll2003/conll2003.dev.conll','../data/conll2003/output_dev_file.conll']
    trainingfile = argv[1]
    inputfile = argv[2]
    outputfile = argv[3]
    
    training_features, gold_labels = extract_features_and_labels(trainingfile)


    for modelname in [
        'logreg', 
        'NB',
        'SVM']:
        print('Tuning:',modelname)
        ml_model, vec = create_classifier(training_features, gold_labels, modelname)

    

# uncomment this when using this in a script    
if __name__ == '__main__':
    # Code below is executed when this python file is called from terminal
    main()
    