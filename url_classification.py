import urllib3
import os
import re
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import spacy

from typing import List, Dict, Tuple, Union, Any
from tqdm import tqdm
import newspaper
from newspaper import Article
from collections import Counter
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
# import datasets
from datasets.dataset_dict import DatasetDict
from datasets import Dataset, load_dataset

import transformers
from transformers import pipeline
from transformers import AutoTokenizer, DataCollatorWithPadding, DistilBertConfig
# from transformers import TFDistilBertModel
from transformers import TFAutoModel, TFAutoModelForSequenceClassification
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

from nltk.tokenize import word_tokenize
from nltk.stem.porter import *
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')
nltk.download('stopwords')


MAIN_FOLDER = '/content/drive/MyDrive/url_classification/'
HTML_FOLDER = MAIN_FOLDER + 'html_files_Nov-24-2022/'
MODELS_FOLDER = MAIN_FOLDER + 'models/'
VARIABLES_FOLDER = MAIN_FOLDER + 'saved_variables/'

URL = 'url'
TEXT = 'text'
LEMMATIZED = 'cleaned_lemmatized_text'
TARGET = 'label'

nlp = spacy.load('en_core_web_sm')

DISTILBERT_MAX_INPUT = 510  # 512 - the [CLS] and [SEP] tokens

labels_encoded = {'Article': 0, 'Blog': 1, 'Event': 2, 'Webinar': 3, 'PR': 4, 'MISC': 5}
labels_decoded = {y: x for x, y in labels_encoded.items()}

num_labels = len(labels_encoded)

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9A-Za-z #+_]')
STOPWORDS = set(stopwords.words('english'))

os.chdir(MAIN_FOLDER)

for path in (MODELS_FOLDER, VARIABLES_FOLDER):
    if not os.path.isdir(path):
        os.makedirs(path)

if not os.path.isdir(HTML_FOLDER):
    raise Exception('HTML folder with relevant files should be already created and located in the main folder.')


def save_variables(variables: Dict[str, Any]) -> None:
    """
    Saves variables to disk using pickle.

    Parameters:
    - variables: dictionary where the keys are the names to use when saving the variables, and the values are the
    variables to be saved.

    Returns:
    - None
    """

    for variable_name, variable in variables.items():
        with open(f'{VARIABLES_FOLDER}{variable_name}.pickle', 'wb') as f:
            pickle.dump(variable, f)


def read_variable(variable_name: str) -> Any:
    """
    Loads a variable previously saved in disk using pickle.

    Parameters:
    - variable_name: path of the variable saved

    Returns:
    - variable: the loaded variable
    """

    with open(f'{VARIABLES_FOLDER}{variable_name}.pickle', 'rb') as f:
        variable = pickle.load(f)

    return variable


def read_csv(csv_path: str, usecols: List[str] = None, namecols: List[str] = None, remove_nan: str = None,
             ignore_dash: bool = False) -> pd.DataFrame:
    """
    Creates a DataFrame from a CSV file.

    Parameters:
    - csv_path: the path to the CSV file.
    - usecols: a list of column names to use from the CSV file. If not provided, all columns will be used.
    - namecols: a list of new names for the columns from the CSV file. If not provided, the original column names will
    be used.
    - remove_nan: the name of the column from where rows with missing values will be removed. It must be from namecols
    (or from usecols, if namecols was not provided). If not provided, all rows will be kept.
    - ignore_dash: a flag indicating whether to ignore rows with a dash (`-`) in the target column. If not provided,
    rows with a dash in the target column will be included in the data.

    Returns:
    - df: a pandas DataFrame object.
    """

    df = pd.read_csv(csv_path, usecols=usecols)

    if namecols:
        assert len(usecols) == len(namecols)
        rename_columns = {usecols[i]: namecols[i] for i in range(len(usecols))}
        df.rename(columns=rename_columns, inplace=True)

    if remove_nan:
        df = df[~df[remove_nan].isna()]

    if ignore_dash:
        df = df[df['label'] != '-']

    # All sections of blogs labeled as 'MISC/Blog?' become part of 'MISC'.
    df.loc[df[TARGET] == 'MISC/Blog?', TARGET] = 'MISC'

    df.reset_index(drop=True, inplace=True)

    return df


def read_htmls(df: pd.DataFrame, column: str) -> List[str]:
    """
    Reads HTML files from a pandas DataFrame.

    Parameters:
    - df: a DataFrame with a column containing the names of the HTML files.
    - column: the name of the column in the DataFrame where the filenames are stored.

    Returns:
    - htmls: a list of strings, one for each HTML file.
    """

    filenames = df[column].values
    htmls = list()

    for i, filename in enumerate(tqdm(filenames)):
        try:
            with open(f'{HTML_FOLDER}{filename}') as f:
                html = f.read()
                htmls.append(html)
        except FileNotFoundError:
            print(f'File {i} not found: "{filename}"')

    return htmls


def read_articles(htmls: List[str]) -> List[newspaper.article.Article]:
    """
    Reads articles from a list of HTML strings.

    Parameters:
    - htmls: a list of HTML strings representing articles.

    Returns:
    - toi_articles: a list of newspaper.article.Article objects, one for each HTML string.
    """

    toi_articles = list()

    for html in tqdm(htmls):
        toi_article = Article(url=' ', language='en')
        toi_article.set_html(html)
        toi_article.parse()
        toi_article.nlp()
        toi_articles.append(toi_article)

    return toi_articles


def create_df_from_articles(df: pd.DataFrame, toi_articles: List[newspaper.article.Article]) -> pd.DataFrame:
    """
    Creates a DataFrame with article text and labels from a list of articles.

    Parameters:
    - df: a DataFrame with a TARGET column containing the labels for the articles.
    - toi_articles: a list of newspaper.article.Article objects.

    Returns: a DataFrame with two columns: TEXT, containing the concatenated title and text of the articles, and TARGET,
    containing the labels for the articles.
    """

    summaries = [(toi_article.title + '. ' + toi_article.text).replace('\n', ' ') for toi_article in toi_articles]
    y = df[TARGET].tolist()
    assert len(y) == len(summaries)

    df_text1 = pd.DataFrame({TEXT: summaries, TARGET: y})

    return df_text1


def read_or_create_variables(variable_names: List[str]) -> List[Any]:
    """
    Reads or creates variables with the given names. If a variable exists, it is read and returned.
    If a variable with a given name does not exist, it is created and an empty list is returned.

    Parameters:
    - variable_names: list of strings representing the names of the variables to be read and/or created.

    Returns:
    - variables: list of the read and/or created variables.
    """

    variables = list()

    for variable_name in variable_names:
        try:
            variable = read_variable(variable_name)
            print(f'SUCCESS: variable {variable_name} was read, it contains {len(variable)} elements.')
        except FileNotFoundError:
            variable = list()
            print(f'Variable {variable_name} was just created and contains {len(variable)} elements.')

        variables.append(variable)

    return variables


def create_new_urls(df: pd.DataFrame, column: str) -> pd.Series:
    """
    Creates the urls to read based on the difference between all URLs and the ones already read.

    Parameters:
    - df: a DataFrame with a column containing the URLs.
    - column: the name of the column in the DataFrame where the URLs are stored.

    Returns:
    - urls_new: pandas Series object with strings of the URLs to read.
    """

    urls = df[column]
    mask = np.isin(urls, urls_old)
    urls_new = urls[~mask]
    print(f'There are {len(urls_new)} new URLs to read.')

    return urls_new


def read_texts_from_urls(urls_new: pd.Series, urls_old: List[str]) -> Tuple[List[str], List[int], List[int]]:
    """
    Reads the relevant text from each URL.

    Parameters:
    - urls_new: pandas Series object with strings of the URLs to read.
    - urls_old: list of strings of the URLs that have already been read.

    Returns a tuple of three elements:
    - texts_new: list of strings of the texts read from the URLs.
    - indexes_new: list of integers representing the indexes of the URLs that were read successfully.
    - idx_label_to_remove: list of integers representing the indexes, starting from 0, that threw errors and should be
    deleted.
    """

    texts_new = list()
    indexes_new = list()

    idx_label_to_remove = list()
    idx_label = 0

    for i, url in enumerate(urls_new, len(urls_old)):
        try:
            toi_article = Article(url=url, language='en')
            toi_article.download()
            toi_article.parse()
            toi_article.nlp()
            text = (toi_article.title + '. ' + toi_article.text).replace('\n', ' ')
            texts_new.append(text)
            indexes_new.append(i)
            print(f'{i}: url "{url}" read successfully.')
        except Exception:
            idx_label_to_remove.append(idx_label)
            print(f'{i}: ERROR: url "{url}" was not read successfully.')

        idx_label += 1

    if idx_label_to_remove:
        print(f'\nWhen reading the new URLs, {len(urls_new) - len(indexes_new)} '
              f'of them threw an error and could not be read.')
    elif texts_new:
        print('All URLs were successfully read.')
    else:
        print('No new URL was read.')

    return texts_new, indexes_new, idx_label_to_remove


def create_new_labels(df: pd.DataFrame, urls_to_read: pd.Series, idx_label_to_remove: List[int]) -> List[str]:
    """
    Creates a list of labels for the given URLs and removes the labels corresponding to the URLs that threw errors.

    Parameters:
    - df: pandas DataFrame containing the labels.
    - urls_to_read: pandas Series object with strings of the URLs for which the labels should be created.
    - idx_label_to_remove: list of integers representing the indexes of the URLs that threw errors and should have their
    labels removed.

    Returns:
    - labels_new: list of labels according to the new URLs, discarding the ones that threw errors.
    """

    labels_new = df.loc[urls_to_read.index, TARGET]
    assert (labels_new == df.loc[labels_new.index, TARGET]).all()
    labels_new.reset_index(drop=True, inplace=True)
    labels_new = labels_new.drop(idx_label_to_remove).tolist()

    return labels_new


def update_variables(old_variables: List[Any], new_variables: List[Any]) -> List[Any]:
    """
    Updates the old variables with the new variables.

    Parameters:
    - old_variables: list of variables that should be updated.
    - new_variables: list of variables used to update the old variables, should be of same length and order as
    `old_variables`.

    Returns:
    - updated_variables: list of variables that are the result of updating the old variables with the new variables.
    """

    assert len(old_variables) == len(new_variables)

    updated_variables = list()
    zipped_variables = list(zip(old_variables, new_variables))

    for old_var, new_var in zipped_variables:
        updated_variable = old_var + new_var
        updated_variables.append(updated_variable)

    return updated_variables


def create_df_from_lists(labels: List[str], indexes: List[int], texts: List[str], urls: List[str]) -> pd.DataFrame:
    """
    Creates a pandas DataFrame with columns for URLs, texts, and labels.

    Parameters:
    - labels: list of strings representing the labels of the articles.
    - indexes: list of integers representing the indexes of the articles.
    - texts: list of strings representing the texts of the articles.
    - urls: list of strings representing the URLs of the articles.

    Returns:
    - df_text2: pandas DataFrame with columns for URLs, texts, and labels.
    """

    df_text2 = pd.DataFrame({
        URL: pd.Series(urls).loc[indexes],
        TEXT: texts,
        TARGET: labels,
    }).reset_index(drop=True)

    return df_text2


def remove_duplicates(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Removes duplicate rows from a DataFrame based on a specific column.

    Parameters:
    - df: a DataFrame with a column containing values to be checked for duplicates.
    - column: the name of the column in the DataFrame where the values are stored.

    Returns:
    - df: DataFrame with duplicate rows removed. If there were no duplicates, the original DataFrame is returned.
    """

    if df[column].duplicated().any():
        original_amount = df.shape[0]
        df = df[~df[column].duplicated()]
        print(f"{original_amount - df.shape[0]} rows had duplicate values in the dataframe's column '{column}' "
              f"and were deleted.")
        assert not df[column].duplicated().any()

    else:
        print(f"There are no duplicate values in the dataframe's column '{column}'.")

    return df


def remove_rows(df: pd.DataFrame, with_errors: bool = False, irrelevant: bool = False,
                below_threshold: int = None) -> pd.DataFrame:
    """
    Removes rows from a pandas DataFrame that meet certain conditions.

    Parameters:
    - df: pandas DataFrame from which rows should be removed.
    - with_errors: boolean flag indicating whether rows with URLs' texts that throw errors should be removed. If set to
    True, rows with texts that have less than 100 words and contain both the strings " 404 " and " error " will be
    removed.
    - irrelevant: boolean flag indicating whether rows with irrelevant labels should be removed. If set to True, rows
    with labels that are not included in the dictionary `labels_encoded` will be removed.
    - below_threshold: integer representing the minimum number of words that the text of a URL must contain for it to be
    included in the final DataFrame. Rows with texts containing fewer words than this threshold will be removed.

    Returns:
    - df: pandas DataFrame with rows that met a condition removed.
    """

    if with_errors:
        words = df[TEXT].str.split().str.len()
        condition1 = words < 100  # empirical threshold
        condition2 = df[TEXT].str.contains(' 404 ')
        condition3 = df[TEXT].str.contains(' error ')
        df = df[~(condition1 & condition2 & condition3)]

    if irrelevant:
        df = df[df[TARGET].isin(labels_encoded)]
        df.reset_index(drop=True, inplace=True)

    # If the words that the text a URL contains is below this threshold, the row is discarded
    if below_threshold:
        words = df[TEXT].str.split().str.len()
        df = df[words >= below_threshold]

    return df


def text_preprocessing(text: str, lemmatize: bool = False, clean: bool = False) -> str:
    """
    Preprocess text by applying lemmatization and cleaning operations.

    Parameters:
    - text: a string of the text to be preprocessed.
    - lemmatize: a flag indicating whether to apply lemmatization to the text. Default is False.
    - clean: a flag indicating whether to clean the text. Default is False.

    Returns:
    - text: a string of the preprocessed text.
    """

    if lemmatize:
        text = ' '.join(token.lemma_ for token in nlp(text))

    if clean:
        text = REPLACE_BY_SPACE_RE.sub(' ', text)
        text = BAD_SYMBOLS_RE.sub('', text)
        text = ' '.join(word for word in text.split() if word not in STOPWORDS)

    return text


def split_data(df: pd.DataFrame, column: str, test_size: float = 0.2, val_size: float = None,
               random_state: int = None) -> Union[Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
                                                  Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series,
                                                        pd.Series]]:
    """
    Splits a pandas DataFrame into training, validation (if needed), and test sets.
    Data is split in a stratified fashion according to `df[TARGET]`.

    Parameters:
    - df: pandas DataFrame containing the data to be split.
    - column: name of the column in the DataFrame to be used as the input data (X).
    - test_size: float representing the proportion of the whole data to be used for the test set. Must be between 0.0
    and 1.0.
    - val_size: float representing the proportion of the whole data to be used for the validation set. Must be between
    0.0 and 1.0.
    - random_state: integer seed for the random number generator.

    Returns:
    - X_train: pandas DataFrame containing the input data for the training set.
    - (if `val_size` is provided) X_val: pandas DataFrame containing the input data for the validation set.
    - X_test: pandas DataFrame containing the input data for the test set.
    - y_train: pandas Series containing the target data for the training set.
    - (if `val_size` is provided) y_val: pandas Series containing the target data for the validation set.
    - y_test: pandas Series containing the target data for the test set.
    """

    X = df[[column]]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_size,
                                                        stratify=y,
                                                        random_state=random_state
                                                        )

    if val_size:
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                          test_size=1 / ((1 - test_size) / val_size),
                                                          stratify=y_train,
                                                          random_state=random_state)

        return X_train, X_val, X_test, y_train, y_val, y_test

    return X_train, X_test, y_train, y_test


def vectorize_data(column: str, X_train: pd.DataFrame, X_test: pd.DataFrame, X_val: pd.DataFrame = None,
                   ngram_range: Tuple[int, int] = (1, 1)) -> Union[
    Tuple[pd.DataFrame, pd.DataFrame, TfidfVectorizer], Tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame, TfidfVectorizer]]:
    """
    Vectorizes the data in a pandas DataFrame column using the TfidfVectorizer.

    Parameters:
    - column: name of the column in the DataFrame to be vectorized.
    - X_train: pandas DataFrame containing the input data for the training set.
    - X_test: pandas DataFrame containing the input data for the test set.
    - X_val: (optional) pandas DataFrame containing the input data for the validation set.
    - ngram_range: tuple of integers specifying the lower and upper boundaries of the range of n-values for different
    n-grams to be extracted.

    Returns:
    - X_train: pandas DataFrame containing the vectorized input data for the training set.
    - (if `X_val` is provided) X_val: pandas DataFrame containing the vectorized input data for the validation set.
    - X_test: pandas DataFrame containing the vectorized input data for the test set.
    - vectorizer: the fitted TfidfVectorizer object.
    """

    vectorizer = TfidfVectorizer(ngram_range=ngram_range)
    X_train_tr = vectorizer.fit_transform(X_train[column])
    X_test_tr = vectorizer.transform(X_test[column])

    if X_val:
        X_val_tr = vectorizer.transform(X_val[column])

        return X_train_tr, X_val_tr, X_test_tr, vectorizer

    return X_train_tr, X_test_tr, vectorizer


def create_dataset_dict(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series,
                        y_test: pd.Series, X_val: pd.DataFrame = None, y_val: pd.Series = None) -> DatasetDict:
    """
    Creates a DatasetDict object from pandas DataFrames and Series.

    Parameters:
    - X_train: pandas DataFrame containing the input data for the training set.
    - X_test: pandas DataFrame containing the input data for the test set.
    - y_train: pandas Series containing the target data for the training set.
    - y_test: pandas Series containing the target data for the test set.
    - X_val: (optional) pandas DataFrame containing the input data for the validation set.
    - y_val: (optional) pandas Series containing the target data for the validation set.

    Returns:
    - dataset: DatasetDict object containing the training, validation (optional) and test sets.
    """

    datasets = {
        'train': Dataset.from_dict(
            {TEXT: X_train[TEXT],
             TARGET: y_train,
             }
        ),
        'test': Dataset.from_dict(
            {TEXT: X_test[TEXT],
             TARGET: y_test,
             }
        )
    }

    if X_val is not None and y_val is not None:
        datasets['validation'] = Dataset.from_dict(
            {TEXT: X_val[TEXT],
             TARGET: y_val,
             }
        )

    dataset = DatasetDict(datasets)

    return dataset


def tokenize(batch: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tokenizes the text data in a batch.

    Parameters:
    - batch: dictionary containing the batch data.

    Returns:
    - batch: dictionary with the text data tokenized.
    """

    tokenized = tokenizer(batch[TEXT], padding=True, truncation=True)

    return tokenized


def get_hidden_states(batch: Dict[str, Any]) -> Dict[str, Any]:
    """
    Gets the hidden states of the input text in a batch.

    Parameters:
    - batch: dictionary containing the batch data.

    Returns:
    - hidden_state: numpy array with the hidden states for the [CLS] tokens of the input texts in the batch.
    """

    # Convert text to tokens
    inputs = tokenizer(
        batch[TEXT],
        padding=True,
        truncation=True,
        return_tensors='tf',
    )

    # Extract last hidden states
    outputs = tf_model(inputs)

    # Return vector for [CLS] token
    return {'hidden_state': outputs.last_hidden_state[:, 0].numpy()}


def fit_model(clf: Any, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
    """
    Fits a classifier model to training data.

    Parameters:
    - clf: a classifier object.
    - X_train: the training data, as a Pandas DataFrame.
    - y_train: the training labels, as a Pandas Series.

    Returns:
    - clf: the fitted classifier object.
    """

    clf.fit(X_train, y_train)

    return clf


def predict(clf: Any, X_test: pd.DataFrame) -> np.ndarray:
    """
    Predicts the labels for test data using a fitted classifier.

    Parameters:
    - clf: a fitted classifier object.
    - X_test: the test data, as a Pandas DataFrame.

    Returns:
    - y_pred: a numpy array with the predicted labels for the test data.
    """

    y_pred = clf.predict(X_test)

    return y_pred


def get_accuracy(y_test: pd.Series, y_pred: np.ndarray) -> float:
    """
    Calculates the accuracy of the predicted labels.

    Parameters:
    - y_test: the true labels for the test data, as a Pandas Series.
    - y_pred: the predicted labels for the test data, as a numpy array.

    Returns:
    - accuracy: the predicted labels' accuracy on the test set
    """

    accuracy = np.mean(y_pred == y_test)

    return accuracy


def print_stratified_kfold(clfs: List[Tuple[str, Any]], X_train: pd.DataFrame, y_train: pd.Series, n_splits: int = 5,
                           cv: int = 5, with_train_val_len_start: bool = False) -> None:
    """
    Prints the results of a Stratified K-Fold cross-validation for a list of classifiers.

    Parameters:
    - clfs: a list of tuples with a string representing the name of the classifier and the classifier object.
    - X_train: the training data, as a Pandas DataFrame.
    - y_train: the training labels, as a Pandas Series.
    - n_splits: the number of splits for the Stratified K-Fold cross-validation. Default is 5.
    - cv: the number of folds for the cross-validation. Default is 5.
    - with_train_val_len_start: flag to indicate whether to call the function print_train_val_len_start and print the
    mean of the training and validation data for every split. Default is False.

    Returns:
    - None
    """

    for clf in clfs:
        print(f'\nStratifiedKFold - classifier: {clf[0]}:\n')
        skf = StratifiedKFold(n_splits=n_splits)

        if with_train_val_len_start:
            for train_index, val_index in skf.split(X_train, y_train):
                print_train_val_len_start(train_index, val_index)
                print(f"\tMean y: TRAIN: {y_train.iloc[train_index].mean():.3f},"
                      f"\tVALIDATION: {y_train.iloc[val_index].mean():.3f}")

        scores = cross_val_score(clf[1],
                                 X_train,
                                 y_train,
                                 cv=cv)

        print_val_scores(scores)


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


def create_tf_dataset(dataset_encoded: DatasetDict, tokenizer: AutoTokenizer, batch_size: int = 16) -> \
        Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Converts a `DatasetDict` object to a tuple of `tf.data.Dataset` objects.

    Parameters:
    - dataset_encoded: DatasetDict object containing datasets with the encoded text data and labels.
    - tokenizer: AutoTokenizer object that will be used to encode the text data.
    - batch_size: integer representing the number of samples per batch.

    Returns:
    - tf_train_dataset, tf_val_dataset, tf_test_dataset: a tuple of tf.data.Dataset objects for the train, validation,
    and test datasets.
    """

    tokenizer_columns = tokenizer.model_input_names

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors='tf')

    tf_train_dataset = dataset_encoded['train'].to_tf_dataset(
        columns=tokenizer_columns,
        label_cols=[TARGET],
        shuffle=True,
        batch_size=batch_size,
        collate_fn=data_collator,
    )

    tf_val_dataset = dataset_encoded['validation'].to_tf_dataset(
        columns=tokenizer_columns,
        label_cols=[TARGET],
        shuffle=False,
        batch_size=batch_size,
        collate_fn=data_collator,
    )

    tf_test_dataset = dataset_encoded['test'].to_tf_dataset(
        columns=tokenizer_columns,
        label_cols=[TARGET],
        shuffle=False,
        batch_size=batch_size,
        collate_fn=data_collator,
    )

    return tf_train_dataset, tf_val_dataset, tf_test_dataset


def create_distilbert_config(dropout: float = 0.1, attention_dropout: float = 0.1) -> transformers.DistilBertConfig:
    """
    Creates a DistilBERT configuration with specified dropout and attention dropout values.

    Parameters:
    - dropout: the dropout rate for the DistilBERT model. Default is 0.1.
    - attention_dropout: the attention dropout rate for the DistilBERT model. Default is 0.1.

    Returns:
    - config: a transformers.DistilBertConfig object representing the configuration of the DistilBERT model.
    """

    config = DistilBertConfig(
        dropout=dropout,
        attention_dropout=attention_dropout,
        output_hidden_states=True,
        num_labels=num_labels,
    )

    return config


def compile_model(tf_model: tf.keras.Model, learning_rate: float = 5e-6) -> tf.keras.Model:
    """
    Compiles a TensorFlow model with Adam optimizer and Sparse Categorical Crossentropy loss.

    Parameters:
    - tf_model: a TensorFlow model to be compiled.
    - learning_rate: the learning rate for the Adam optimizer. Default is 5e-6.

    Returns:
    - tf_model: the compiled TensorFlow model.
    """

    tf_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=tf.metrics.SparseCategoricalAccuracy()
    )

    return tf_model


def train_model(tf_model: tf.keras.Model, tf_train_dataset: tf.data.Dataset, tf_val_dataset: tf.data.Dataset,
                epochs: int = 100, patience: int = 3) -> tf.keras.Model:
    """
    Trains a TensorFlow model using a training and validation datasets, with early stopping.

    Parameters:
    - tf_model: a compiled TensorFlow model to be trained.
    - tf_train_dataset: a TensorFlow dataset with the training data.
    - tf_val_dataset: a TensorFlow dataset with the validation data.
    - epochs: the number of epochs to train the model. Default is 100.
    - patience: the number of epochs to wait before stopping training if validation loss doesn't improve. Default is 3.

    Returns:
    - tf_model: the trained TensorFlow model.

    """

    callback = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True
    )

    tf_model.fit(
        tf_train_dataset,
        validation_data=tf_val_dataset,
        epochs=epochs,
        callbacks=[callback]
    )

    return tf_model


def check_if_exists(variable_name: str):
    """
    Checks if a variable exists in the global scope.

    Parameters:
    - variable_name: name of the variable

    Returns:
    - None
    """

    if variable_name in globals():
        print(f'Variable "{variable_name}" exists.')
    else:
        print(f'Variable "{variable_name}" does not exist.')


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


def get_best_clf(clfs: List[Tuple[str, Any]], X_train: pd.DataFrame, X_test: pd.Series, y_train: pd.DataFrame,
                 y_test: pd.Series) -> Tuple[Any, str, float]:
    """
    Finds the classifier with the highest accuracy score on the test data.

    Parameters:
    - clfs: list of tuples containing the names and instances of the classifiers to be evaluated.
    - X_train: numpy array with the input features for the training set.
    - X_test: numpy array with the input features for the test set.
    - y_train: numpy array with the target labels for the training set.
    - y_test: numpy array with the target labels for the test set.

    Returns:
    - best_clf: instance of the classifier with the highest accuracy score on the test data.
    - best_clf_name: name of the classifier with the highest accuracy score on the test data.
    - best_accuracy: float with the highest accuracy score on the test data.
    """

    results = list()
    best_clf = None
    best_clf_name = None
    best_accuracy = 0

    for clf_name, clf in clfs:
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)

        if acc > best_accuracy:
            best_accuracy = acc
            best_clf = clf
            best_clf_name = clf_name

        results.append({'clf': clf_name,
                        'accuracy': acc})

    print(pd.DataFrame(results).set_index('clf'))
    return best_clf, best_clf_name, best_accuracy


def print_train_val_len_start(train_index: List[int], val_index: List[int]) -> None:
    """
    Prints the length and starting indices of the training and validation sets.

    Parameters:
    - train_index: list of integers representing the indices of the training set.
    - val_index: list of integers representing the indices of the validation set.

    Returns:
    - None
    """

    print(f'TRAIN index len: {len(train_index)}, start: {train_index[:3]}, '
          f'\tVALIDATION index len: {len(val_index)}, start: {val_index[:3]}')


def print_val_scores(scores: List[float], extra_info: bool = False) -> None:
    """
    Prints the mean and all the scores of a cross-validation.

    Parameters:
    - scores: list of float values representing the scores of the cross-validation.
    - extra_info: (optional) boolean value indicating whether to print the standard deviation, minimum and maximum
    values of the scores.

    Returns:
    - None
    """

    print(f'Cross validation scores: mean: {np.mean(scores):.3f}, all: {[round(score, 3) for score in scores]}')
    if extra_info:
        print(f'(std: {np.std(scores):.3f}, min: {min(scores):.3f}, max: {max(scores):.3f})')


def print_confusion_matrix(clf_name: str, y_test: List[int], y_pred: List[int], with_report: bool = False) -> None:
    """
    Prints a confusion matrix and (optional) a classification report for a given classifier.

    Parameters:
    - clf_name: string containing the name of the classifier.
    - y_test: list of integers with the correct labels of the test set.
    - y_pred: list of integers with the predicted labels of the test set.
    - with_report: bool indicating if a classification report should be printed.

    Returns:
    - None.
    """

    accuracy = get_accuracy(y_test, y_pred)

    y_test = [labels_decoded[x] for x in y_test]
    y_pred = [labels_decoded[x] for x in y_pred]

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title(f'{clf_name} - acc {accuracy:.3f}', size=15)
    plt.show()

    if with_report:
        print('\n' + classification_report(y_test, y_pred))


if '__name__' == '__main__':
    df1 = read_csv('activities_unlabeled.csv',
                   usecols=['File Name', 'Label'],
                   namecols=['filename', 'label'],
                   remove_nan='filename',
                   ignore_dash=True)

    htmls = read_htmls(df1, 'filename')

    toi_articles = read_articles(htmls)

    df_text1 = create_df_from_articles(df1, toi_articles)

    save_variables(variables={'df_text1': df_text1})

    df2 = read_csv('activities_labeled13.csv',
                   usecols=['url', 'true_label'],
                   namecols=['url', 'label'],
                   remove_nan='label')

    labels_old, indexes_old, texts_old, urls_old = read_or_create_variables(
        ['labels_old', 'indexes_old', 'texts_old', 'urls_old'])

    urls_new = create_new_urls(df2, 'url')

    texts_new, indexes_new, idx_label_to_remove = read_texts_from_urls(urls_new, urls_old)

    labels_new = create_new_labels(df2, urls_new, idx_label_to_remove)

    labels, indexes, texts, urls = update_variables(old_variables=[labels_old, indexes_old, texts_old, urls_old],
                                                    new_variables=[labels_new, indexes_new, texts_new,
                                                                   urls_new.tolist()])

    assert len(texts) == len(pd.Series(urls).loc[indexes]) == len(indexes) == len(labels)

    df_text2 = create_df_from_lists(labels, indexes, texts, urls)

    save_variables(variables={'labels_old': labels,
                              'indexes_old': indexes,
                              'texts_old': texts,
                              'urls_old': urls,
                              'df_text2': df_text2})

    df_text2 = remove_duplicates(df_text2, URL)

    df_text = pd.concat([df_text1, df_text2]).reset_index(drop=True)

    df_text = remove_duplicates(df_text, TEXT)

    df_text = remove_rows(df_text, with_errors=True, irrelevant=True, below_threshold=51)

    df_text[LEMMATIZED] = df_text[TEXT].apply(lambda x: text_preprocessing(x, lemmatize=True, clean=True))

    save_variables({'df_text': df_text})

    # 1. BoW approach

    ## Preprocessing

    df_text[TARGET] = df_text[TARGET].replace(labels_encoded)

    X_train, X_test, y_train, y_test = split_data(df_text, LEMMATIZED, test_size=0.2, random_state=0)

    X_train_tr, X_test_tr, vectorizer = vectorize_data(LEMMATIZED, X_train, X_test, ngram_range=(1, 1))

    ## Modeling

    clfs = [
        ('LogisticRegression', LogisticRegression(max_iter=3000,
                                                  class_weight='balanced')
         ),
        ('RandomForest', RandomForestClassifier(max_depth=18,
                                                n_estimators=75,
                                                random_state=0)
         ),
        ('KNN 5', KNeighborsClassifier(n_neighbors=5)
         ),
        ('SVM C1', SVC(C=1,
                       class_weight='balanced')
         ),
        ('MultinomialNB', MultinomialNB()
         ),
    ]

    print_stratified_kfold(clfs, X_train_tr, y_train)

    clf, clf_name, test_acc = get_best_clf(clfs, X_train_tr, X_test_tr, y_train, y_test)
    print(f'Best classifier: {clf_name}, test accuracy: {test_acc:.3f}')

    # Chosen model

    clf = fit_model(LogisticRegression(max_iter=3000,
                                       class_weight='balanced',
                                       ),
                    X_train_tr,
                    y_train,
                    )

    y_pred = predict(clf, X_test_tr)
    y_probs = clf.predict_proba(X_test_tr)

    print_confusion_matrix('Logistic Regression', y_test, y_pred, with_report=True)

    pickle.dump(clf, open(f'{MODELS_FOLDER}bow_lr_clf', 'wb'))

    df_mistakes = create_df_mistakes(df_text, LEMMATIZED, X_test, y_test, y_pred, y_probs)

    save_variables({'df_mistakes_bow_lr_clf': df_mistakes})

    plot_distribution_of_confidences(y_test, y_pred, y_probs,
                                     print_statistical_measures=True)

    # 2. DistilBERT approach + ML

    ## Preprocessing

    distilbert_model = 'distilbert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(distilbert_model)

    tf_model = TFAutoModel.from_pretrained(distilbert_model)

    df_text[TARGET] = df_text[TARGET].replace(labels_encoded)

    ## Extra preprocessing for ML approach

    X_train, X_test, y_train, y_test = split_data(df_text, column=TEXT, test_size=0.2, val_size=None, random_state=0)

    dataset = create_dataset_dict(X_train, X_test, y_train, y_test)

    dataset_encoded = dataset.map(
        tokenize,
        batched=True,
        batch_size=None,
    )

    dataset_encoded.reset_format()

    dataset_hidden = dataset_encoded.map(
        get_hidden_states,
        batched=True,
        batch_size=16,
    )

    X_train_hidden = np.array(dataset_hidden['train']['hidden_state'])
    y_train_hidden = np.array(dataset_hidden['train'][TARGET])

    X_test_hidden = np.array(dataset_hidden['test']['hidden_state'])
    y_test_hidden = np.array(dataset_hidden['test'][TARGET])

    save_variables({'X_train_hidden': X_train_hidden,
                    'X_test_hidden': X_test_hidden,
                    'y_train_hidden': y_train_hidden,
                    'y_test_hidden': y_test_hidden,
                    })

    clfs = [
        ('LogisticRegression', LogisticRegression(max_iter=3000,
                                                  class_weight='balanced')
         ),
        ('RandomForest', RandomForestClassifier(max_depth=18,
                                                n_estimators=75,
                                                random_state=0)
         ),
        ('KNN 5', KNeighborsClassifier(n_neighbors=5)
         ),
        ('SVM C1', SVC(C=1,
                       class_weight='balanced')
         ),
    ]

    print_stratified_kfold(clfs, X_train_hidden, y_train)

    lr_clf = fit_model(LogisticRegression(max_iter=3000,
                                          class_weight='balanced',
                                          ),
                       X_train_hidden,
                       y_train_hidden)

    y_pred = predict(lr_clf, X_test_hidden)
    y_probs = lr_clf.predict_proba(X_test_hidden)

    print_confusion_matrix('Logistic Regression', y_test_hidden, y_pred, with_report=True)

    pickle.dump(lr_clf, open(f'{MODELS_FOLDER}distilbert_lr_clf', 'wb'))

    df_mistakes = create_df_mistakes(df_text, TEXT, X_test, y_test, y_pred, y_probs)

    save_variables({'df_mistakes_distilbert_lr_clf': df_mistakes})

    plot_distribution_of_confidences(y_test, y_pred, y_probs, print_statistical_measures=True)

    # 3. DistilBERT approach + DL

    ## Extra preprocessing

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_text, column=TEXT, test_size=0.2, val_size=0.1,
                                                                random_state=0)

    dataset = create_dataset_dict(X_train, X_test, y_train, y_test, X_val, y_val)

    dataset_encoded = dataset.map(
        tokenize,
        batched=True,
        batch_size=None,
    )

    tf_train_dataset, tf_val_dataset, tf_test_dataset = create_tf_dataset(dataset_encoded, tokenizer)

    config = create_distilbert_config(dropout=0.1, attention_dropout=0.1)

    tf_model = (TFAutoModelForSequenceClassification.from_pretrained(
        distilbert_model,
        config=config,
    )
    )

    tf_model = compile_model(tf_model, learning_rate=2e-6)

    tf_model = train_model(tf_model, tf_train_dataset, tf_val_dataset, epochs=1000, patience=5)

    tf.keras.models.save_model(
        tf_model,
        filepath=f'{MODELS_FOLDER}distilbert_nn',
        overwrite=True,
        save_format='tf'
    )

    loss, test_accuracy = tf_model.evaluate(tf_test_dataset)
    print("Loss: {}\t Test Accuracy: {}".format(loss, test_accuracy))

    output_logits = tf_model.predict(tf_test_dataset).logits
    y_pred = np.argmax(output_logits, axis=-1)
    y_probs = tf.nn.softmax(output_logits).numpy()

    print_confusion_matrix('Deep Learning', y_test, y_pred, with_report=True)

    df_mistakes = create_df_mistakes(df_text, TEXT, X_test, y_test, y_pred, y_probs)

    plot_distribution_of_confidences(y_test, y_pred, y_probs)

    save_variables({'df_mistakes_distilbert_nn': df_mistakes})
