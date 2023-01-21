from constants import MODELS_FOLDER, TARGET, TEXT, labels_encoded
from variables import save_variables, read_variable
# from modeling import tokenizer, tf_model

from preprocessing import split_data, create_dataset_dict, tokenize, get_hidden_states
from eda import create_df_mistakes, plot_distribution_of_confidences
from modeling import print_stratified_kfold, print_confusion_matrix

import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

df_text = read_variable('df_text')


def main():
    """
    Main function of the modules that fine-tunes the DistilBERT pre-trained model with Logistic Regression,
    reports a confusion matrix, plots the distributions of confidence, and saves the relevant model and variables.

    Parameters:
    - None

    Returns:
    - None
    """

    # Preprocessing

    df_text[TARGET] = df_text[TARGET].replace(labels_encoded)

    # Extra preprocessing

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

    # Modeling

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

    lr_clf = LogisticRegression(max_iter=3000,
                                class_weight='balanced',
                                )

    lr_clf.fit(X_train_hidden, y_train_hidden)

    y_pred = lr_clf.predict(X_test_hidden)
    y_probs = lr_clf.predict_proba(X_test_hidden)

    print_confusion_matrix('Logistic Regression', y_test_hidden, y_pred, with_report=True)

    pickle.dump(lr_clf, open(f'{MODELS_FOLDER}distilbert_lr_clf', 'wb'))

    df_mistakes = create_df_mistakes(df_text, TEXT, X_test, y_test, y_pred, y_probs)

    save_variables({'df_mistakes_distilbert_lr_clf': df_mistakes})

    plot_distribution_of_confidences(y_test, y_pred, y_probs, print_statistical_measures=True)


if __name__ == '__main__':
    main()
