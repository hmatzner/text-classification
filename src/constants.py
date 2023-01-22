from labels import labels_encoded

MAIN_FOLDER = '/Users/hernanmatzner/text_classification/'
DATA_FOLDER = MAIN_FOLDER + 'data/'
HTML_FOLDER = MAIN_FOLDER + 'html_files/'
VARIABLES_FOLDER = MAIN_FOLDER + 'saved_variables/'
MODELS_FOLDER = MAIN_FOLDER + 'models/'

URL = 'url'
TEXT = 'text'
LEMMATIZED = 'cleaned_lemmatized_text'
TARGET = 'label'

labels_decoded = {y: x for x, y in labels_encoded.items()}
num_labels = len(labels_encoded)
