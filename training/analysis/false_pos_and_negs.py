import pandas as pd

def create_false_pos_and_negs_csv(results_frame: pd.DataFrame):
    true_pos_frame = results_frame[(results_frame['labels_remodelled_yes_no'] == 1) & \
                                    (results_frame['pred_labels'] == 1)]
    true_neg_frame = results_frame[(results_frame['labels_remodelled_yes_no'] == 0) & \
                                    (results_frame['pred_labels'] == 0)]
    false_pos_frame = results_frame[(results_frame['labels_remodelled_yes_no'] == 0) & \
                                    (results_frame['pred_labels'] == 1)]
    false_neg_frame = results_frame[(results_frame['labels_remodelled_yes_no'] == 1) & \
                                    (results_frame['pred_labels'] == 0)]    
    
    return true_pos_frame, true_neg_frame, false_pos_frame, false_neg_frame