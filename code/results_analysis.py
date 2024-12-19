period_list=[]
hyphen_list=[]
digit_list=[]
upper_list=[]
lower_list=[]
title_list=[]
other_list=[]

error_period=0
error_hyphen=0
error_capital=0
error_upper=0
error_lower=0
error_title=0
error_digit=0
error_other=0

no_error_period = 0
no_error_hyphen = 0
no_error_capital = 0
no_error_upper=0
no_error_lower=0
no_error_title=0
no_error_digit=0
no_error_other=0
with open('../data/classified_data/model_predictions/output_SVM_no.conll','r', encoding='utf8') as infile:
	for line in infile:
            components = line.rstrip('\n').split()
            if len(components) > 4:
                token = components[0]
                true_label = components[-2]
                predicted_label = components[-1]
                shape = ''
                if true_label != predicted_label:
                    for char in token:
                        if char == '.':
                            shape = 'Has_period'
                            error_period+=1
                            period_list.append([token,true_label,predicted_label])
                        elif char =='-':
                            shape = 'Has_hyphen'
                            error_hyphen+=1
                            hyphen_list.append([token,true_label,predicted_label])
                    if token.isupper():
                        shape = shape + 'Upper'
                        error_capital+=1
                        error_upper+=1
                        upper_list.append([token,true_label,predicted_label])
                    elif token.islower():
                        shape = shape + 'Lower'
                        error_capital+=1
                        error_lower+=1
                        lower_list.append([token,true_label,predicted_label])
                    elif token.istitle():
                        shape = shape + 'First_letter_capital'
                        error_capital+=1
                        error_title+=1
                        title_list.append([token,true_label,predicted_label])
                    elif token.isdigit():
                        shape = shape + 'Digit'
                        error_capital+=1
                        error_digit+=1
                        digit_list.append([token,true_label,predicted_label])
                    else:
                        shape = 'Other'
                        error_capital+=1
                        error_other+=1
                        other_list.append([token,true_label,predicted_label])
                
                if true_label == predicted_label:
                    for char in token:
                        if char == '.':
                            shape = 'Has_period'
                            no_error_period+=1
                        elif char =='-':
                            shape = 'Has_hyphen'
                            no_error_hyphen+=1
                    if token.isupper():
                        shape = shape + 'Upper'
                        no_error_capital+=1
                        no_error_upper+=1
                    elif token.islower():
                        shape = shape + 'Lower'
                        no_error_capital+=1
                        no_error_lower+=1
                    elif token.istitle():
                        shape = shape + 'First_letter_capital'
                        no_error_capital+=1
                        no_error_title+=1
                    elif token.isdigit():
                        shape = shape + 'Digit'
                        no_error_capital+=1
                        no_error_digit+=1
                    else:
                        shape = 'Other'
                        no_error_capital+=1
                        no_error_other+=1
                
print('errors with period:',error_period, 'no error with period',no_error_period) 
print('errors with hyphen:',error_hyphen,'no error with hyphen',no_error_hyphen,'\n')    
print('errors with capital:',error_capital,'no error with capital',no_error_capital)    
print('errors with upper:',error_upper,'no error with upper',no_error_upper)   
print('errors with lower:',error_lower,'no error with lower',no_error_lower)   
print('errors with title:',error_title,'no error with title',no_error_title)  
print('errors with digit:',error_digit,'no error with digit',no_error_digit)   
print('errors with other:',error_other,'no error with other',no_error_other)   

with open('../data/classified_data/mistakes/period_mistakes_SVM.txt','w') as outfile:
    outfile.write(str(period_list))
with open('../data/classified_data/mistakes/hyphen_mistakes_SVM.txt','w') as outfile:
    outfile.write(str(hyphen_list))
with open('../data/classified_data/mistakes/digit_mistakes_SVM.txt','w') as outfile:
    outfile.write(str(digit_list))
with open('../data/classified_data/mistakes/upper_mistakes_SVM.txt','w') as outfile:
    outfile.write(str(upper_list))
with open('../data/classified_data/mistakes/lower_mistakes_SVM.txt','w') as outfile:
    outfile.write(str(lower_list))
with open('../data/classified_data/mistakes/title_mistakes_SVM.txt','w') as outfile:
    outfile.write(str(title_list))
with open('../data/classified_data/mistakes/other_mistakes_SVM.txt','w') as outfile:
    outfile.write(str(other_list))


