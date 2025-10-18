import numpy as np
import pandas as pd
from classes.scaler import Scaler
from imblearn.over_sampling import SMOTE
from classes.evaluation import compute_metrics, classes_report
from models.svm import SVM, SVMKernel
from models.lr import LogisticRegression, LogisticRegressionKernel

def hinge_loss(y_true, margin_pred):
    """
    Compute hinge loss for the evaluation of model.
    :param y_true: true label;
    :param margin_pred: predicted label.
    :return: result of hinge loss.
    """
    return np.mean(np.maximum(0, 1 - y_true * margin_pred))

def binary_log_loss(y_true, y_prob, eps=1e-15):
    """
    Compute binary log loss for the evaluation of model.
    :param y_true: true label;
    :param y_prob: predicted label;
    :param eps: coefficient used to avoid computational problems.
    :return: binary logistic loss.
    """
    y_true = np.array(y_true)
    # management of classes -1 | +1
    if set(np.unique(y_true)).issubset({-1, 1}):
        y_true = (y_true + 1) // 2
    # clip to avoid computational problems
    y_prob = np.clip(y_prob, eps, 1 - eps)
    return - np.mean(y_true * np.log2(y_prob) + (1 - y_true) * np.log2(1 - y_prob))

def preprocessing(X_train, y_train, X_test, random_state=42):
    """
    Apply preprocessing to data in order to prepare them for training phase.
    :param X_train: set of feature vectors for training;
    :param y_train: label of feature vectors for training;
    :param X_test: set of feature vectors for test;
    :param random_state: random state for reproducibility (standard set to 42);
    :return: processed and augmented data.

    Preprocessing phase is done by fulfill the following steps:
    1 - logarithmic transformation of data to reduce distances between feature vectors;
    2 - application of scaling;
    3 - application of SMOTE for data augmentation (only on the training set).
    """
    # logarithmic transformation of data
    X_train_log = np.log(X_train + 1)
    X_test_log = np.log(X_test + 1)
    # scaling of data
    scaler = Scaler()
    X_train_scaled = scaler.fit_transform(X_train_log)
    X_test_scaled = scaler.transform(X_test_log)
    # application of SMOTE for data augmentation (applied only on training data)
    smote = SMOTE(random_state=random_state)
    X_train_aug, y_train_aug = smote.fit_resample(X_train_scaled, y_train)
    return X_train_aug, X_test_scaled, y_train_aug

def get_folds(indices, n_folds):
    """
    Obtain indices for folds.
    :param indices: array of indices;
    :param n_folds: number of folds wanted.
    :return: list of elements for each fold.
    """
    # obtain size of folds
    fold_size = int(len(indices) / n_folds)
    # remainder necessary to handle remaining indices after division
    remainder = len(indices) - fold_size * n_folds
    # list of folds
    all_folds = []
    start_index = 0
    # iterate for number of folds
    for i in range(n_folds):
        # compute end of index for each fold
        end_index = start_index + fold_size
        # handling of remaining indices
        if i < remainder:
            # ending index increased by one
            end_index += 1
        # list of indices for i-th fold
        fold_indices = indices[start_index:end_index]
        # append to main list
        all_folds.append(fold_indices)
        # next fold will start at the end of the previous
        start_index = end_index
    return all_folds

def stratified_k_fold_indices(y, n_folds, random_state=42):
    """
    Obtain stratified indices for folds.
    :param y: labels.
    :param n_folds: number of folds.
    :param random_state: random state for reproducibility (standard set to 42).
    :return: stratified k-fold indices.
    """
    # conversion to numpy array
    y = np.array(y)
    # find classes
    unique_classes = np.unique(y)
    folds_per_class = {cls: [] for cls in unique_classes}
    # dictionary that associate class -> indices
    indices_per_class = {cls: np.where(y == cls)[0] for cls in unique_classes}
    # random number generator using random_state
    rng = np.random.RandomState(random_state)
    # shuffle applied for each unique class
    for cls in unique_classes:
        rng.shuffle(indices_per_class[cls])
        # split in folds
        folds_per_class[cls] = get_folds(indices_per_class[cls], n_folds)

    stratified_folds = []
    # build final folds
    for fold_idx in range(n_folds):
        fold_indices = []
        for cls in unique_classes:
            fold_indices.extend(folds_per_class[cls][fold_idx])
        stratified_folds.append(np.array(fold_indices))
    return stratified_folds

def nested_k_fold_cross_validation(X, y, models, n_outer_folds=5, n_inner_folds=4, random_state_cv=42):
    """
    Perform nested k-fold cross validation to evaluate performance of different models.
    :param X: feature set;
    :param y: label set;
    :param models: models and their grid-search for evaluation;
    :param n_outer_folds: number of outer folds;
    :param n_inner_folds: number of inner folds;
    :param random_state_cv: random state of cross validation employed for reproducibility.
    :return: results of metrics of nested k-fold cross validation.
    """
    results = {}

    for model_name, config in models.items():
        print(f"=== {model_name} ===")
        # extract class
        modelClass = config['class']
        # extract parameters from grid
        param_grid = config['param_grid']
        # extract random state from parameters (applied to model)
        # if not inserted, use 42
        random_state = config.get('random_state', random_state_cv)

        # training metrics
        train_accuracy_per_outer_fold = []
        train_precision_per_outer_fold = []
        train_recall_per_outer_fold = []
        train_f1_per_outer_fold = []
        training_error_per_outer_fold = []

        # test metrics
        test_accuracy_per_outer_fold = []
        test_precision_per_outer_fold = []
        test_recall_per_outer_fold = []
        test_f1_per_outer_fold = []
        test_error_per_outer_fold = []

        # errors on outer fold
        oof_errors_list = []

        # loss curves for each model on outer fold
        loss_curves_for_model = []

        # metrics for class
        class_metrics = []

        # list of the best hyperparameters for each outer fold
        best_hyperparams_per_fold = []

        # obtain outer folds (used for test)
        outer_folds = stratified_k_fold_indices(y, n_outer_folds, random_state=random_state)

        # loop on outer folds
        for i, test_outer_indices in enumerate(outer_folds):

            print(f"=== Outer Fold {i + 1} ===")

            # obtain indices for training
            train_outer_indices = np.concatenate([f for j, f in enumerate(outer_folds) if j != i])
            X_train_outer, y_train_outer = X[train_outer_indices], y[train_outer_indices]
            X_test_outer, y_test_outer = X[test_outer_indices], y[test_outer_indices]

            best_hyperparams = None
            best_loss = np.inf

            # set random state of stratified k fold function +1 each time
            inner_random_state = random_state + i
            inner_folds = stratified_k_fold_indices(y_train_outer, n_inner_folds, random_state=inner_random_state)

            param_keys = list(param_grid.keys())

            # method used to generate combinations of parameters
            # necessary for grid search
            def generate_combinations(params, index):
                if index == len(param_keys):
                    return [params.copy()]
                key = param_keys[index]
                combinations = []
                for value in param_grid[key]:
                    params[key] = value
                    combinations.extend(generate_combinations(params, index + 1))
                return combinations

            # generation of all parameter's combinations
            param_combinations = generate_combinations({}, 0)

            # loop for each combination created
            for params in param_combinations:
                # memorize inner loss
                inner_losses = []

                for j, test_inner_indices in enumerate(inner_folds):
                    # obtain samples for training
                    train_inner_indices = np.concatenate([f for k, f in enumerate(inner_folds) if k != j])
                    X_train_inner, y_train_inner = X_train_outer[train_inner_indices], y_train_outer[train_inner_indices]
                    X_test_inner, y_test_inner = X_train_outer[test_inner_indices], y_train_outer[test_inner_indices]
                    # apply pre-processing of data (+ data augmentation)
                    X_train_inner_scaled, X_test_inner_scaled, y_train_inner_aug = preprocessing(
                        X_train_inner, y_train_inner, X_test_inner, random_state=inner_random_state)
                    # apply random state to model
                    model_params = params.copy()
                    model_params['random_state'] = inner_random_state
                    model = modelClass(**model_params)
                    # training
                    model.fit(X_train_inner_scaled, y_train_inner_aug)
                    # if model is SVM / SVM Kernel Gaussian / SVM Kernel Polynomial
                    # apply hinge loss
                    if isinstance(model, SVM) or isinstance(model, SVMKernel):
                        margins = model.predict(X_test_inner_scaled, margins=True)
                        loss = hinge_loss(y_test_inner, margins)

                    # else if model is LR / LR Kernel Gaussian / LR Kernel Polynomial
                    # apply binary log loss
                    elif isinstance(model, LogisticRegression) or isinstance(model, LogisticRegressionKernel):
                        y_proba = model.predict_proba(X_test_inner_scaled)
                        loss = binary_log_loss(y_test_inner, y_proba)

                    else:
                        raise ValueError("Model not supported for loss.")
                    # save inner loss
                    inner_losses.append(loss)
                # if loss is better than others, save hyperparameters
                avg_inner_loss = np.mean(inner_losses)
                if avg_inner_loss < best_loss:
                    best_loss = avg_inner_loss
                    best_hyperparams = params

            print(f"Best hyperparams: {best_hyperparams} in outer fold {i + 1}")

            best_hyperparams_per_fold.append(best_hyperparams)

            # preprocessing outer fold with inner random state
            X_train_outer_scaled, X_test_outer_scaled, y_train_outer_aug = preprocessing(
                X_train_outer, y_train_outer, X_test_outer, random_state=inner_random_state)
            # setting final model with the best hyperparameters found
            final_model_params = best_hyperparams.copy()
            final_model_params['random_state'] = inner_random_state
            best_model = modelClass(**final_model_params)
            best_model.fit(X_train_outer_scaled, y_train_outer_aug)
            # save loss of model
            loss_curves_for_model.append(best_model.loss_history)

            # predictions
            y_pred_outer = best_model.predict(X_test_outer_scaled)
            y_pred_train_outer = best_model.predict(X_train_outer_scaled)

            # determination of errors
            error_mask = y_pred_outer != y_test_outer
            error_indices = test_outer_indices[error_mask]
            error_preds = y_pred_outer[error_mask]
            error_trues = y_test_outer[error_mask]

            # save errors
            for idx, true_label, pred_label in zip(error_indices, error_trues, error_preds):
                oof_errors_list.append({
                    'sample_index': idx,
                    'true_label': true_label,
                    'predicted_label': pred_label,
                    'outer_fold': i + 1
                })

            # compute metrics
            train_acc, train_prec, train_rec, train_f1 = compute_metrics(y_train_outer_aug, y_pred_train_outer)
            test_acc, test_prec, test_rec, test_f1 = compute_metrics(y_test_outer, y_pred_outer)
            report_outer = classes_report(y_test_outer, y_pred_outer)
            test_error = np.sum(y_test_outer != y_pred_outer) / len(y_test_outer)
            training_error = np.sum(y_pred_train_outer != y_train_outer_aug) / len(y_train_outer_aug)

            # save metrics
            train_accuracy_per_outer_fold.append(train_acc)
            train_precision_per_outer_fold.append(train_prec)
            train_recall_per_outer_fold.append(train_rec)
            train_f1_per_outer_fold.append(train_f1)
            training_error_per_outer_fold.append(training_error)

            test_accuracy_per_outer_fold.append(test_acc)
            test_precision_per_outer_fold.append(test_prec)
            test_recall_per_outer_fold.append(test_rec)
            test_f1_per_outer_fold.append(test_f1)
            test_error_per_outer_fold.append(test_error)

            class_metrics.append(report_outer)

            print(f"[Outer Fold {i + 1}] Accuracy: {test_acc:.4f}, Precision: {test_prec:.4f}, "
                  f"Recall: {test_rec:.4f}, F1 score: {test_f1:.4f}")

        errors_df = pd.DataFrame(oof_errors_list)

        results[model_name] = {
            'train_accuracy': np.mean(train_accuracy_per_outer_fold),
            'train_accuracy_std': np.std(train_accuracy_per_outer_fold),
            'train_precision': np.mean(train_precision_per_outer_fold),
            'train_precision_std': np.std(train_precision_per_outer_fold),
            'train_recall': np.mean(train_recall_per_outer_fold),
            'train_recall_std': np.std(train_recall_per_outer_fold),
            'train_f1_score': np.mean(train_f1_per_outer_fold),
            'train_f1_score_std': np.std(train_f1_per_outer_fold),
            'test_accuracy': np.mean(test_accuracy_per_outer_fold),
            'test_accuracy_std': np.std(test_accuracy_per_outer_fold),
            'test_precision': np.mean(test_precision_per_outer_fold),
            'test_precision_std': np.std(test_precision_per_outer_fold),
            'test_recall': np.mean(test_recall_per_outer_fold),
            'test_recall_std': np.std(test_recall_per_outer_fold),
            'test_f1_score': np.mean(test_f1_per_outer_fold),
            'test_f1_score_std': np.std(test_f1_per_outer_fold),
            'oof_errors': errors_df,
            'class_metrics': class_metrics,
            'test_error': np.mean(test_error_per_outer_fold),
            'test_error_std': np.std(test_error_per_outer_fold),
            'training_error': np.mean(training_error_per_outer_fold),
            'training_error_std': np.std(training_error_per_outer_fold),
            'hyperparameters': best_hyperparams_per_fold,
            'loss_curves': loss_curves_for_model,
        }

    return results
