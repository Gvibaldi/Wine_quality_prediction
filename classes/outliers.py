import numpy as np

def count_outliers(dataset, column):
    """
    Determine the number of outliers present in a column of the dataset.
    :param dataset: the dataset considered;
    :param column: the column of the dataset considered.
    :return: the number of outliers of the column.
    """

    # compute first percentile Q1 (25%)
    Q1 = np.percentile(dataset[column], 25)
    # compute third percentile Q3 (75%)
    Q3 = np.percentile(dataset[column], 75)
    # compute InterQuantile Range (IQR) defined by the difference between Q3 and Q1
    IQR = Q3 - Q1
    # compute lower and upper bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # determine the number of outliers by considering the bounds
    num_outliers = dataset[(dataset[column] < lower_bound) | (dataset[column] > upper_bound)].shape[0]
    # return the number of outliers
    return num_outliers