import numpy as np

def get_performance_metrics(gt, pred):
    gt_np =  np.asarray(gt, dtype=np.int)
    pred_np = np.asarray(pred, dtype=np.int)
    precision = np.sum(gt_np * pred_np) / (np.sum(pred_np) + 1e-8)
    recall = np.sum(gt_np * pred_np) / (np.sum(gt_np) + 1e-8)
    accuracy = (np.sum(gt_np == pred_np) / len(gt))
    return accuracy, precision, recall
