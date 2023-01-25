from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
import pandas 
import matplotlib.pyplot as plt
import csv
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn

def acc_score(x,y, tag):  
    accuracy = accuracy_score(x,y)
    print(f'trenētā modeļa {tag} taga precizitāte: {accuracy}')
        
#draws heatmap of selected categories
def draw_heatmap(x_labels, y_labels):
    class_names = ['ADJ','ADP','ADV','AUX','CCONJ','DET','INTJ','NOUN','NUM','PART','PRON','PROPN','PUNCT','SCONJ','SYM','VERB','X']
    cm = confusion_matrix(x_labels, y_labels, labels=class_names)

    seaborn.heatmap(cm, annot=True, vmax=200, fmt='g',
                xticklabels=class_names, 
                yticklabels=class_names,
                annot_kws={"size": 15})
    seaborn.set(font_scale=20)
    plt.xlabel("Paredzētā vērtība", fontsize=16)
    plt.ylabel("Patiesā vērtība",fontsize=16)
    plt.show()

#creates file with the incorrect predictions
def print_false(x_labels, y_labels):
    tokens = []
    false = []
    correct = []
    for token, prediction, label in zip(x_df['token'].values.tolist(), y_labels, x_labels):
        if prediction != label:
                tokens.append(token)
                false.append(prediction)
                correct.append(label)

    prediction_df = pd.DataFrame()
    prediction_df['token'] = tokens
    prediction_df['predicted_label'] = false
    prediction_df['true_label'] = correct

    prediction_df.to_csv(f'predictions_flair.csv', header=False, index=False, sep='\t', mode='w',  encoding='utf-8')



x_path = 'udt211_test_full.txt'
y_path = 'results/flair_pos_tags.txt'

x_columns = ['token','lemma','pos','morph','morph_full']
#change columns based on the formatting of the model output file
y_columns = ['token','pos']

x_df = pandas.read_csv(x_path, encoding='utf-8',sep='\t', header=None,names=x_columns)
y_df = pandas.read_csv(y_path, encoding='utf-8', sep='\t', header=None,names=y_columns, quoting=csv.QUOTE_NONE)
#use it if model doesnt use _ for empty tags like it is in UD
#y_df.loc[y_df["morph_full"].isnull(), "morph_full"] = '_'

x_labels = x_df['pos']
y_labels = y_df['pos']
acc_score(x_labels,y_labels,'pos')

