from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.feature_extraction import DictVectorizer
import pandas as pd
import seaborn as sn
import numpy as np
import sys
import string
from sklearn.metrics import classification_report,confusion_matrix, ConfusionMatrixDisplay, recall_score, f1_score,precision_score,  accuracy_score
from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors
from nltk.stem import WordNetLemmatizer
import pickle
lemmatizer = WordNetLemmatizer()

    
def extract_features_and_labels(trainingfile, feature):
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
    previous_token = ''
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
               
                if feature == 'token':
                    feature_dict = {'pos': pos, 'lemma': lemma, 
                                'shape': shape,'previous token':previous_token}
                    
                elif feature == 'pos':
                    feature_dict = {'token': token, 'lemma': lemma, 
                                'shape': shape,'previous token':previous_token}
                    
                elif feature == 'lemma':
                    feature_dict = {'token': token, 'pos': pos, 
                                'shape': shape,'previous token':previous_token}
                    
                elif feature == 'shape':
                    feature_dict = {'token': token, 'pos': pos, 
                                'lemma': lemma,'previous token':previous_token}
                    
                elif feature == 'no token and no lemma':
                    feature_dict = {'pos': pos, 'shape': shape,'previous token':previous_token}
                    
                elif feature == 'pos and shape':
                    feature_dict = {'token':token, 'lemma':lemma, 'previous token':previous_token}
                    
                elif feature == 'previous token':
                    feature_dict = {'token':token, 'pos': pos, 'lemma': lemma, 
                                'shape': shape}
                else :
                    feature_dict = {'token':token, 'pos': pos, 'lemma': lemma, 
                                'shape': shape, 'previous token':previous_token}
                    
                data.append(feature_dict)
                targets.append(components[-1])
                previous_token = token
    
    return data, targets
    
def extract_features_and_labels_from_output(output_file):
    """
    Extracts ground truth and predicted labels from the output file.

    This function processes a file that contains the original data along with 
    the predicted labels, and extracts the ground truth (true) labels and 
    predicted labels. The function assumes that the input file has the format 
    where the ground truth label is the second-to-last element on each line, 
    and the predicted label is the last element.

    Args:
        output_file: The path to the output file containing the data, 
                           ground truth labels, and predicted labels. 
                           The file is expected to have the format:
                           <data> <ground_truth_label> <predicted_label>
    
    Returns:
        tuple: A tuple containing:
            - `gt_labels`: A list of the ground truth labels.
            - `pred_labels`: A list of the predicted labels.
    """
    gt_labels = []
    pred_labels = []
    with open(output_file, 'r', encoding='utf8') as infile:
        for line in infile:
            components = line.rstrip('\n').split()
            if len(components) > 0:
                gt_labels.append(components[-2])
                pred_labels.append(components[-1])
    return gt_labels, pred_labels
    

def extract_features(trainingfile,feature):
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
              
                if feature == 'token':
                    feature_dict = {'pos': pos, 'lemma': lemma, 
                                'shape': shape,'previous token':previous_token}
                    
                elif feature == 'pos':
                    feature_dict = {'token': token, 'lemma': lemma, 
                                'shape': shape,'previous token':previous_token}
                    
                elif feature == 'lemma':
                    feature_dict = {'token': token, 'pos': pos, 
                                'shape': shape,'previous token':previous_token}
                    
                elif feature == 'shape':
                    feature_dict = {'token': token, 'pos': pos, 
                                'lemma': lemma,'previous token':previous_token}
                    
                elif feature == 'no token and no lemma':
                    feature_dict = {'pos': pos, 'shape': shape,'previous token':previous_token}
                    
                elif feature == 'pos and shape':
                    feature_dict = {'token':token, 'lemma':lemma, 'previous token':previous_token}
                    
                elif feature == 'previous token':
                    feature_dict = {'token':token, 'pos': pos, 'lemma': lemma, 
                                'shape': shape}
                
                else :
                    feature_dict = {'token':token, 'pos': pos, 'lemma': lemma, 
                                'shape': shape, 'previous token':previous_token}
                    
                data.append(feature_dict)
                previous_token = token
    
             
    return data
    

def evaluate(output_file, label = None):
    """
    Evaluates the performance of a predictive model by comparing its predictions 
    to the ground truth labels using various metrics.

    This function calculates the following evaluation metrics:
    - Recall
    - Precision
    - F1-Score
    - Accuracy

    Additionally, it computes and optionally displays the confusion matrix if the `label` 
    parameter is provided. The confusion matrix shows how the predicted labels 
    compare to the actual labels.

    Arguments:
        ground_truth:       A list of the true labels.
        predict_label:      A list of the predicted labels.
        label (optional):   A list of possible labels for classification. 
                            This parameter is used to order the confusion matrix. 
                            If not provided, the confusion matrix is computed without specific label order.

    Returns:
        tuple: A tuple containing the following elements:
            - `recall`: The recall score, which is the ratio of correctly predicted positive observations 
                                to all actual positives (weighted average, other settings are possible).
            - `precision`: The precision score, which is the ratio of correctly predicted positive observations 
                                   to the total predicted positives (weighted average, other settings are possible).
            - `f_score`: The F1 score, which is the weighted average of precision and recall (weighted average, other settings are possible).
            - `accuracy`: The accuracy score, which is the ratio of correctly predicted labels to all predictions.
            - `display_cf`: A plot of the confusion matrix 

    """
    ground_truth, predict_label = extract_features_and_labels_from_output(output_file)

    recall = recall_score(ground_truth,predict_label, average ='macro')
    precision = precision_score(ground_truth,predict_label, average = 'macro')
    f_score = f1_score(ground_truth,predict_label, average = 'macro')
    accuracy = accuracy_score(ground_truth,predict_label)

    cf_matrix = confusion_matrix(ground_truth,predict_label,labels = label)
    display_cf = ConfusionMatrixDisplay(confusion_matrix = cf_matrix, display_labels = label)
 
    results = f'''
    the recall is: {recall}
    the precision is: {precision}
    the f-score is: {f_score}
    the accuracy is: {accuracy}
    the confusionmatrix is:
{cf_matrix}'''
    return results




def create_classifier(train_features, train_targets, feature, modelname = 'logreg'):
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
    
    with open('../models/model_parameters/best_parameterslogreg','r') as infile:
        content = infile.read()
        content = content.strip('[]').split()
        c = content[2]
        c = float(c.strip(','))
        penalty = content[4]
        penalty = penalty.strip("'}")
    model = LogisticRegression(max_iter=2000, C=c,penalty=penalty)
    features_combined = features_vectorized
 
       
            
    model.fit(features_combined, train_targets)
    
    with open('../../models/feature_ablation_study'+ modelname+feature +'.pkl','wb') as model_file:
            pickle.dump(model,model_file)
    return model, vec
    
def classify_data(model, vec, inputdata, outputfile, modelname,feature):
    """
    Classifies input data using a trained model and saves the predictions to an output file.

    This function extracts features from the input data, vectorizes the features using
    the provided vectorizer (`vec`), makes predictions using the trained model (`model`),
    and writes the original data along with the corresponding predictions to the specified output file.

    Arguments:
        model: A trained classifier model (in this case logistic regression) used to make predictions.
        vec: A fitted `DictVectorizer` used to transform the input features into the same format 
             as the model's training data.
        inputdata: The path to the input file containing data to be classified. The file should 
                         contain tokenized text or structured data, depending on the feature extraction.
        outputfile: The path to the output file where the predictions will be saved. 
                          Each line in the output will contain the original input data followed by its predicted label.

    Returns:
        The function writes the results directly to the `outputfile`. No return value.
    """
  
    
    features = extract_features(inputdata,feature)
    features_combined = vec.transform(features)
    
    predictions = model.predict(features_combined)
    outfile = open(outputfile, 'w')
    counter = 0
    
    for line in open(inputdata, 'r'):
        if len(line.rstrip('\n').split()) > 0:
            outfile.write(line.rstrip('\n') + '\t' + predictions[counter] + '\n')
            counter += 1
    outfile.close()
    counter=0
    with open('../data/classified_data/feature_ablation_study/output_without_'+feature+'.conll','w') as outgoingfile:
        for line in open(inputdata, 'r'):
            if len(line.rstrip('\n').split()) > 0:
                outgoingfile.write(line.rstrip('\n') + '\t' + predictions[counter] + '\n')
                counter +=1

    
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
        
    #Note 1: argv[0] is the name of the python program if you run your program as: python program1.py arg1 arg2 arg3
    #Note 2: sys.argv is simple, but gets messy if you need it for anything else than basic scenarios with few arguments
    #you'll want to move to something better. e.g. argparse (easy to find online)
    
    #data_folder = "../../data/conll2003/"

    #train_file = data_folder + "conll2003.train.conll"
    #test_file = data_folder + "conll2003.test.conll"
    #dev_file = data_folder + "conll2003.dev.conll"
    #you can replace the values for these with paths to the appropriate files for now, e.g. by specifying values in argv
    argv =['classification_script','../../data/conll2003/conll2003.train.conll','../../data/conll2003/conll2003.dev.conll','../../data/conll2003/output_dev_file.conll']
    trainingfile = argv[1]
    inputfile = argv[2]
    outputfile = argv[3]
    feature_dict =['token', 'pos', 'lemma','shape','no token and no lemma', 'pos and shape', 'previous token', 'baseline with all features']
    for feature in feature_dict: 
        for modelname in ['logreg']:
            print(modelname,'model ran without this feature:',feature)
    
            
            training_features, gold_labels = extract_features_and_labels(trainingfile,feature)
            
            ml_model, vec = create_classifier(training_features, gold_labels,feature)
                
            classify_data(ml_model, vec, inputfile, outputfile,modelname, feature)
            
            evaluation = evaluate(outputfile)
            print(evaluation)
            print()
            print('---------------')
    

# uncomment this when using this in a script    
if __name__ == '__main__':
    # Code below is executed when this python file is called from terminal
    main()
    