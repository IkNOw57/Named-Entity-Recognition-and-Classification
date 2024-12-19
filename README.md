Code contains:
- A script called 'tuning.py' which optimizes the classifier based on RandomizedSearchCV, the optimal settings are then written to '../models/model_parameters' for each model.
- A script called 'feature_ablation_study.py', in which the performance of the proposed features are compared. This script prints the output of the model without the feature mentioned.
- A script called 'classification_task.py, which contains the training and evaluation of the classifiers. The output is the evaluation of the model on the classification task, the trained model as a pickle file in '../models/', and the result of the model in a text file.
- A script called 'results_analysis.py, which contains the code to analyse any of the output files created by classification_task.py. The output is an overview of the error distribution and a file containing all instances of mistakes.
- 'BERT-finetunen.py', a notebook in which 3 different seeds of BERT were fine tuned on the Named Entity Recognition task.

Data contains:
- classified_data that resulted from the classification task
- conll2003, the training, development and test files used for the classification task

Models:
- model parameters: the settings optimized for each model
- feature ablation study: the model as a pickle without the feature mentioned in the name
- all models as pickle file
- Google News word embeddings (GoogleNews-vectors-negative300.bin.gz)