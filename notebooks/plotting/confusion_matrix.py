import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sn
import numpy as np
import matplotlib.pyplot as plt

def compute_true_false_positives_and_negatives(results_frame: pd.DataFrame):
    
    true_pos = results_frame.loc[(results_frame['labels_remodelled_yes_no'] == 1) & \
                                    (results_frame['pred_labels'] == 1)]
    true_neg = results_frame.loc[(results_frame['labels_remodelled_yes_no'] == 0) & \
                                    (results_frame['pred_labels'] == 0)]
    false_pos = results_frame.loc[(results_frame['labels_remodelled_yes_no'] == 0) & \
                                    (results_frame['pred_labels'] == 1)]
    false_neg = results_frame.loc[(results_frame['labels_remodelled_yes_no'] == 1) & \
                                    (results_frame['pred_labels'] == 0)]
    
    return true_pos, true_neg, false_pos, false_neg

def compute_confusion_matrix(true_pos,
                                true_neg,
                                false_pos,
                                false_neg):
    true_pos_count = len(true_pos)
    true_neg_count = len(true_neg)
    false_pos_count = len(false_pos)
    false_neg_count = len(false_neg)

    print(f'{true_pos_count=}')
    print(f'{true_neg_count=}')
    print (f'{false_pos_count=}')
    print(f'{false_neg_count=}')

    return true_pos_count, true_neg_count, false_pos_count, false_neg_count

def compute_and_plot_confusion_matrix(results_frame: pd.DataFrame):
    # TODO — https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea
    
    cm = confusion_matrix(y_true=results_frame['labels_remodelled_yes_no'].tolist(),
                          y_pred=results_frame['pred_labels'].tolist())

    group_names = ['True Neg','False Pos','False Neg','True Pos']
    group_counts = ["{0:0.0f}".format(value) for value in
                cm.flatten()]
    
    group_percentages = ["{0:.2%}".format(value) for value in
                     cm.flatten()/np.sum(cm)]

    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
    
    labels = np.asarray(labels).reshape(2,2)



    classes = ['no_remodeling', 'needs_remodeling']
    df_cm = pd.DataFrame(cm, index = [i for i in classes],
                  columns = [i for i in classes])

    plt.figure()
    sn.heatmap(df_cm, annot=labels, fmt='', cmap='Blues')


    return cm

def read_residue_name(residue: str):
    res_name = residue[-4:-1]
    return res_name

def classify_residues(results_frame: pd.DataFrame):
    results_frame['residue_type'] = list(map(read_residue_name, results_frame['input_residue_name']))

    return results_frame

