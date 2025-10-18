import numpy as np

def compute_metrics(y_true, y_pred):
    """
    Compute Accuracy, Precision, Recall and F1 score.
    :param y_true: real labels;
    :param y_pred: predicted labels.
    :return: accuracy, precision, recall and f1-score.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("Length of arrays must be the same")

    # conversion to numpy array
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # compute True Positive (TN), True Negative (TN), False Positive (FP), False Negative (FN)
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == -1) & (y_pred == -1))
    FP = np.sum((y_true == -1) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == -1))
    # accuracy:  (TP + TN) / all real true labels
    accuracy = (TP + TN) / len(y_true) if len(y_true) > 0 else 0
    # precision: (TP / (TP + FP))
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    # recall: (TP / (TP + FN))
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    # compute F1 score (harmonic mean of precision and recall)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return [accuracy, precision, recall, f1]

def classes_report(y_true, y_pred):
    """
    Compute Precision, Recall and F1 score for each class.
    :param y_true: real labels;
    :param y_pred: predicted labels.
    :return: print report of predictions per class.
    """

    # conversion to numpy array
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # classes used
    classes = [-1, 1]
    # save report
    report = {}

    # for each class
    for cls in classes:
        # true positive
        TP = np.sum((y_true == cls) & (y_pred == cls))
        # false Positive
        FP = np.sum((y_true != cls) & (y_pred == cls))
        # false Negative
        FN = np.sum((y_true == cls) & (y_pred != cls))
        # precision: (TP / (TP + FP))
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        # recall: (TP / (TP + FN))
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        # compute F1 score (harmonic mean of precision and recall)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        support = np.sum(y_true == cls)

        report[cls] = {
            'precision': precision,
            'recall': recall,
            'f1-score': f1,
            'support': support
        }
    return report


def print_results(report):
    """
    Print a complete report.
    :param report: data of nested k-fold cross validation.
    :return: print metrics for evaluation.
    """
    reports = report['class_metrics']

    metrics = {-1: {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
               1: {'precision': [], 'recall': [], 'f1-score': [], 'support': []}}

    for fold_report in reports:
        for cls in [-1, 1]:
            for metric in ['precision', 'recall', 'f1-score', 'support']:
                metrics[cls][metric].append(fold_report[cls][metric])

    print("==================== TRAINING METRICS ======================")
    print(f"Accuracy: {report['train_accuracy'] * 100:.2f}%, std: {report['train_accuracy_std'] * 100:.2f}%")
    print(f"Precision: {report['train_precision'] * 100:.2f}%, std: {report['train_precision_std'] * 100:.2f}%")
    print(f"Recall: {report['train_recall'] * 100:.2f}%, std: {report['train_recall_std'] * 100:.2f}%")
    print(f"F1 score: {report['train_f1_score'] * 100:.2f}%, std: {report['train_f1_score_std'] * 100:.2f}%")
    print("====================== TEST METRICS ========================")
    print(f"Accuracy: {report['test_accuracy'] * 100:.2f}%, std: {report['test_accuracy_std'] * 100:.2f}%")
    print(f"Precision: {report['test_precision'] * 100:.2f}%, std: {report['test_precision_std'] * 100:.2f}%")
    print(f"Recall: {report['test_recall'] * 100:.2f}%, std: {report['test_recall_std'] * 100:.2f}%")
    print(f"F1 score: {report['test_f1_score'] * 100:.2f}%, std: {report['test_f1_score_std'] * 100:.2f}%")
    print("================ TRAINING AND TEST ERROR ===================")
    print(f"Training error: {report['training_error']:.2f}, std: {report['training_error_std']:.2f}%")
    print(f"Test error: {report['test_error']:.2f}, std: {report['test_error_std']:.2f}%")
    print("================== BEST HYPERPARAMETERS ====================")
    for hype in report['hyperparameters']:
        print(f"{hype}")
    print("==================== RESULTS FOR CLASSES ===================")
    print(f"Precision +1: {np.mean(metrics[+1]['precision']) * 100:.2f}%, std: {np.std(metrics[+1]['precision']) * 100:.2f}%")
    print(f"Precision -1: {np.mean(metrics[-1]['precision']) * 100:.2f}%, std: {np.std(metrics[-1]['precision']) * 100:.2f}%")
    print(f"Recall +1: {np.mean(metrics[+1]['recall']) * 100:.2f}%, std: {np.std(metrics[+1]['recall']) * 100:.2f}%")
    print(f"Recall -1: {np.mean(metrics[-1]['recall']) * 100:.2f}%, std: {np.std(metrics[-1]['recall']) * 100:.2f}%")
    print(f"F1 score +1: {np.mean(metrics[+1]['f1-score']) * 100:.2f}%, std: {np.std(metrics[+1]['f1-score']) * 100:.2f}%")
    print(f"F1 score -1: {np.mean(metrics[-1]['f1-score']) * 100:.2f}%, std: {np.std(metrics[-1]['f1-score']) * 100:.2f}%")
    print(f"Support +1: {np.mean(metrics[+1]['support']):.0f}")
    print(f"Support -1: {np.mean(metrics[-1]['support']):.0f}")
