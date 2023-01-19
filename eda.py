import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, Any
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize

URL = 'url'

labels_encoded = {'Article': 0, 'Blog': 1, 'Event': 2, 'Webinar': 3, 'PR': 4, 'MISC': 5}
labels_decoded = {y: x for x, y in labels_encoded.items()}


def get_most_common_tokens(df: pd.DataFrame, column: str, amount: int = 10):
    """
    Get the most common tokens in a given column of a dataframe.

    Parameters:
    - df (pandas.DataFrame): Dataframe containing the column.
    - column: column name of the pandas Dataframe.
    - amount: Amount of most common tokens to return.

    Returns:
    - most_common: a list of the most common elements and their counts from the most common to the least
    """

    tokenized = [word_tokenize(string) for string in df[column]]
    flat_list = [item for sublist in tokenized for item in sublist]
    most_common = Counter(flat_list).most_common(amount)

    return most_common


def print_important_features(clf: Any, vectorizer: TfidfVectorizer, features: int = 5) -> None:
    """
    Prints the most important features of a classifier when using a linear kernel.

    Parameters:
    - clf: the trained classifier object.
    - vectorizer: the fitted TfidfVectorizer object.
    - features: the number of most important features to print. Default: 5.

    Returns:
    - None
    """
    coef_importances = np.argsort(clf.coef_, axis=1)

    for i, class_ in enumerate(coef_importances[:, :features]):
        print(f'Class "{labels_decoded[clf.classes_[i]]}" - {features} most important features: '
              f'{vectorizer.get_feature_names_out()[class_]}')


def create_df_mistakes(df_text: pd.DataFrame, column: str, X_test: pd.DataFrame, y_test: pd.Series,
                       y_pred: Union[np.ndarray, pd.Series], y_probs: np.ndarray) -> pd.DataFrame:
    """
    Creates a pandas DataFrame with the rows from `X_test` that have been misclassified and their respective true and
    predicted labels, along with the confidences of each of those predictions.
    The DataFrame also includes the URLs from `df_text` that correspond to the rows of `X_test` that have been
    misclassified.

    Parameters:
    - df_text: pandas DataFrame that contains the URLs of the samples in `X_test`.
    - column: string that represents the name of the column in `df_text` and `X_test` that contains the indices that
    link both DataFrames.
    - X_test: pandas DataFrame that contains the features of the test set.
    - y_test: pandas Series that contains the true labels of the test set.
    - y_pred: NumPy array or pandas Series with the predicted labels of the test set.
    - y_probs: NumPy array with the probabilities of each label in `y_pred`.

    Returns:
    - df_mistakes: pandas DataFrame detailed in the function explanation
    """

    if type(y_test) == pd.core.series.Series:
        y_pred = pd.Series(y_pred, index=y_test.index)  # setting y_pred to same type and indexes as y_test

    elif type(y_test) == np.ndarray:
        y_pred = pd.Series(y_pred)
        y_test = pd.Series(y_test)

    mask = y_pred != y_test

    df = X_test.copy()[mask]
    df['y_true'] = y_test[mask].replace(labels_decoded)
    df['y_pred'] = y_pred[mask].replace(labels_decoded)

    assert (df['y_true'] != df['y_pred']).all()

    df_mistakes = pd.merge(df, df_text[[URL, column]], on=column)
    df_mistakes.index = df.index

    df_confidences = df_mistakes[['y_true', 'y_pred']].applymap(lambda x: labels_encoded[x])

    confidence_pred = y_probs[mask, df_confidences['y_pred']]
    confidence_true = y_probs[mask, df_confidences['y_true']]

    df_mistakes['conf_true'] = confidence_true.round(2)
    df_mistakes['conf_pred'] = confidence_pred.round(2)

    df_mistakes = df_mistakes[[URL, column, 'y_true',
                               'conf_true', 'y_pred', 'conf_pred']]

    return df_mistakes


def plot_distribution_of_confidences(y_test: pd.Series, y_pred: np.ndarray, y_probs: np.ndarray,
                                     print_statistical_measures: bool = False) -> None:
    """
    Plots the distribution of confidence scores for correctly and incorrectly classified samples.

    Parameters:
    - y_test: numpy array of shape (n_samples,) containing the true labels for the test set.
    - y_pred: numpy array of shape (n_samples,) containing the predicted labels for the test set.
    - y_probs: numpy array of shape (n_samples, n_classes) containing the predicted probabilities for each class in the
    test set.
    - print_statistical_measures: boolean flag indicating whether to also print the median and mean confidence scores
    for correctly and incorrectly classified samples. Default value is False.

    Returns:
    - None
    """

    sns.set_theme()

    mask = y_test != y_pred

    wrong_conf_pred = np.max(y_probs[mask], axis=1)
    right_conf_pred = np.max(y_probs[~mask], axis=1)
    assert y_probs.shape[0] == wrong_conf_pred.shape[0] + right_conf_pred.shape[0]

    if print_statistical_measures:
        print(
            f'Confidence of incorrectly classified samples \t- Median: {np.median(wrong_conf_pred):.4f}, '
            f'Mean: {np.mean(wrong_conf_pred):.4f}.')
        print(
            f'Confidence of correctly classified samples \t- Median: {np.median(right_conf_pred):.4f}, '
            f'Mean: {np.mean(right_conf_pred):.4f}.\n')

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

    ax[0, 0].hist(wrong_conf_pred, bins=np.linspace(0, 1, 11), density=None, color='r', alpha=0.6)
    ax[0, 0].set_title('Incorrectly classified samples', size=16)
    ax[0, 0].set_xlabel('Confidence')
    ax[0, 0].set_ylabel('Number of samples')

    ax[0, 1].hist(right_conf_pred, bins=np.linspace(0, 1, 11), density=None, color='g', alpha=0.6)
    ax[0, 1].set_title('Correctly classified samples', size=16)
    ax[0, 1].set_xlabel('Confidence')
    ax[0, 1].set_ylabel('Number of samples')

    ax[1, 0].hist(wrong_conf_pred, bins=np.linspace(0, 1, 11), density=True, color='r', alpha=0.6, cumulative=1)
    ax[1, 0].set_xlabel('Confidence')
    ax[1, 0].set_ylabel('Cumulative distribution')

    ax[1, 1].hist(right_conf_pred, bins=np.linspace(0, 1, 11), density=True, color='g', alpha=0.6, cumulative=1)
    ax[1, 1].set_xlabel('Confidence')
    ax[1, 1].set_ylabel('Cumulative distribution')

    plt.tight_layout()
    plt.show()

    sns.reset_orig()
