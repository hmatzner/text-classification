from variables import save_variables, read_variable
from constants import MODELS_FOLDER, TARGET, LEMMATIZED, labels_encoded
from preprocessing import vectorize_data, split_data
from eda import create_df_mistakes, plot_distribution_of_confidences
from modeling import print_stratified_kfold, get_best_clf, print_confusion_matrix

import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB

df_text = read_variable('df_text')


def main():
    """
    Main function of the modules that trains a Logistic Regression model on Bag of Words with TF-IDF vectorizer,
    reports a confusion matrix, plots the distributions of confidence, and saves the relevant model and variables.

    Parameters:
    - None

    Returns:
    - None
    """

    # Preprocessing

    df_text[TARGET] = df_text[TARGET].replace(labels_encoded)

    X_train, X_test, y_train, y_test = split_data(
        df_text,
        LEMMATIZED,
        test_size=0.2,
        random_state=0)

    X_train_tr, X_test_tr, vectorizer = vectorize_data(LEMMATIZED, X_train, X_test, ngram_range=(1, 1))

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
        ('MultinomialNB', MultinomialNB()
         ),
    ]

    print_stratified_kfold(clfs, X_train_tr, y_train)

    clf, clf_name, test_acc = get_best_clf(clfs, X_train_tr, X_test_tr, y_train, y_test)
    print(f'Best classifier: {clf_name}, test accuracy: {test_acc:.3f}')

    # Chosen model

    lr_clf = LogisticRegression(max_iter=3000,
                                class_weight='balanced',
                                )

    lr_clf.fit(X_train_tr, y_train)

    y_pred = lr_clf.predict(X_test_tr)
    y_probs = lr_clf.predict_proba(X_test_tr)

    print_confusion_matrix('Logistic Regression', y_test, y_pred, with_report=True)

    pickle.dump(clf, open(f'{MODELS_FOLDER}bow_lr_clf', 'wb'))

    df_mistakes = create_df_mistakes(df_text, LEMMATIZED, X_test, y_test, y_pred, y_probs)

    save_variables({'df_mistakes_bow_lr_clf': df_mistakes})

    plot_distribution_of_confidences(y_test, y_pred, y_probs,
                                     print_statistical_measures=True)


if __name__ == '__main__':
    main()
