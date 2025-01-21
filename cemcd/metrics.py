import sklearn.metrics
import numpy as np

def calculate_concept_accuracies(c_pred, c_true):
    c_pred = c_pred.cpu().detach().numpy()
    c_true = c_true.cpu().detach().numpy()

    c_accuracies = []
    c_aucs = []
    overall_auc = 0
    overall_accuracy = 0
    num_seen = 0

    for i in range(c_pred.shape[-1]):
        truth = c_true[:, i]
        indices = np.logical_or(truth == 0, truth == 1).astype(bool)
        if not np.any(indices):
            c_accuracies.append(float("nan"))
            continue
        num_seen += 1
        truth = truth[indices]
        prediction = c_pred[:, i][indices]
        accuracy = sklearn.metrics.accuracy_score(truth, prediction > 0.5)
        c_accuracies.append(accuracy)
        overall_accuracy += accuracy
        if len(np.unique(truth)) == 1:
            auc = sklearn.metrics.accuracy_score(truth,  prediction > 0.5)
        else:
            auc = sklearn.metrics.roc_auc_score(
                truth,
                prediction
            )
        c_aucs.append(auc)
        overall_auc += auc

    num_seen = num_seen if num_seen > 0 else 1
    overall_accuracy /= num_seen
    overall_auc /= num_seen
    return overall_accuracy, c_accuracies, overall_auc, c_aucs

def calculate_task_accuracy(y_pred, y_true):
    if len(y_pred.shape) < 2 or y_pred.shape[-1] == 1:
        y_probs = y_pred.cpu().detach().numpy()
        y_pred = y_probs > 0.5
    else:
        y_pred = y_pred.argmax(dim=-1).cpu().detach().numpy()

    y_true = y_true.cpu().detach().numpy()

    return sklearn.metrics.accuracy_score(y_true, y_pred)
