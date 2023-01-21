from constants import MODELS_FOLDER, TARGET, TEXT, labels_encoded
from variables import save_variables, read_variable
from modeling import tokenizer
from preprocessing import split_data, create_dataset_dict, tokenize
from eda import create_df_mistakes, plot_distribution_of_confidences
import modeling
from modeling import print_confusion_matrix, create_tf_dataset, create_distilbert_config, compile_model, train_model

import numpy as np
import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer, TFAutoModel

df_text = read_variable('df_text')


def main():
    """
    Main function of the modules that fine-tunes the DistilBERT pre-trained model with Deep Learning,
    reports a confusion matrix, plots the distributions of confidence, and saves the relevant model and variables

    Parameters:
    - None

    Returns:
    - None
    """

    # Preprocessing

    df_text[TARGET] = df_text[TARGET].replace(labels_encoded)

    # Extra preprocessing

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_text, column=TEXT, test_size=0.2, val_size=0.1,
                                                                random_state=0)

    dataset = create_dataset_dict(X_train, X_test, y_train, y_test, X_val, y_val)

    dataset_encoded = dataset.map(
        tokenize,
        batched=True,
        batch_size=None,
    )

    tf_train_dataset, tf_val_dataset, tf_test_dataset = create_tf_dataset(dataset_encoded, tokenizer)

    # Modeling

    config = create_distilbert_config(dropout=0.1, attention_dropout=0.1)

    tf_model = (TFAutoModelForSequenceClassification.from_pretrained(
        modeling.distilbert_model,
        config=config,
    )
    )

    tf_model = compile_model(tf_model, learning_rate=2e-6)

    tf_model = train_model(tf_model, tf_train_dataset, tf_val_dataset, epochs=1000, patience=2)

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


if __name__ == '__main__':
    main()
