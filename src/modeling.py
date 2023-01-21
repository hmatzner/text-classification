from constants import TARGET, labels_decoded, num_labels

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Any, Union
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from datasets.dataset_dict import DatasetDict
import transformers
from transformers import AutoTokenizer, DataCollatorWithPadding, DistilBertConfig, TFAutoModel
import tensorflow as tf

distilbert_model = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(distilbert_model)
tf_model = TFAutoModel.from_pretrained(distilbert_model)


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


def print_stratified_kfold(clfs: List[Tuple[str, Any]], X_train: Union[pd.DataFrame, np.ndarray],
                           y_train: pd.Series, n_splits: int = 5,
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


def print_confusion_matrix(clf_name: str, y_test: List[int], y_pred: Union[List[int], np.ndarray],
                           with_report: bool = False) -> None:
    """
    Prints a confusion matrix and (optional) a classification report for a given classifier.

    Parameters:
    - clf_name: string containing the name of the classifier.
    - y_test: list of integers with the correct labels of the test set.
    - y_pred: numpy array of integers with the predicted labels of the test set.
    - with_report: bool indicating if a classification report should be printed.

    Returns:
    - None.
    """

    accuracy = np.mean(y_pred == y_test)

    y_test = [labels_decoded[x] for x in y_test]
    y_pred = [labels_decoded[x] for x in y_pred]

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title(f'{clf_name} - acc {accuracy:.3f}', size=15)
    plt.show()

    if with_report:
        print('\n' + classification_report(y_test, y_pred))


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


# def get_accuracy(y_test: pd.Series, y_pred: np.ndarray) -> float:
#     """
#     Calculates the accuracy of the predicted labels.
#
#     Parameters:
#     - y_test: the true labels for the test data, as a Pandas Series.
#     - y_pred: the predicted labels for the test data, as a numpy array.
#
#     Returns:
#     - accuracy: the predicted labels' accuracy on the test set
#     """
#
#     accuracy = np.mean(y_pred == y_test)
#
#     return accuracy


def create_tf_dataset(dataset_encoded: DatasetDict, batch_size: int = 16) -> \
        Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Converts a `DatasetDict` object to a tuple of `tf.data.Dataset` objects.

    Parameters:
    - dataset_encoded: DatasetDict object containing datasets with the encoded text data and labels.
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


def compile_model(learning_rate: float = 5e-6) -> tf.keras.Model:
    """
    Compiles a TensorFlow model with Adam optimizer and Sparse Categorical Crossentropy loss.

    Parameters:
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

    callback = tf.keras.callbacks.EarlyStopping(
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
