from typing import Dict, Tuple, Union, Any
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from datasets.dataset_dict import DatasetDict
from datasets import Dataset
from transformers import AutoTokenizer, TFAutoModel

URL = 'url'
TEXT = 'text'
LEMMATIZED = 'cleaned_lemmatized_text'
TARGET = 'label'

distilbert_model = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(distilbert_model)
tf_model = TFAutoModel.from_pretrained(distilbert_model)


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
