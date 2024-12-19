#open and execute all scripts in the code folder.
print('Executing Tuning Script')
exec(open('tuning.py').read())
print('Executing Feature Ablation Script')
exec(open('feature_ablation_study.py').read())
print('Executing Classification Script')
exec(open('classification_script.py').read())
print('Executing Results Analysis Script')
exec(open('results_analysis.py').read())