import os

MAIN_FOLDER = '/Users/hernanmatzner/BrewProject/'
DATA_FOLDER = MAIN_FOLDER + 'data/'
HTML_FOLDER = MAIN_FOLDER + 'html_files_Nov-24-2022/'
VARIABLES_FOLDER = MAIN_FOLDER + 'saved_variables/'
MODELS_FOLDER = MAIN_FOLDER + 'models/'

URL = 'url'
TEXT = 'text'
LEMMATIZED = 'cleaned_lemmatized_text'
TARGET = 'label'

labels_encoded = {'Article': 0, 'Blog': 1, 'Event': 2, 'Webinar': 3, 'PR': 4, 'MISC': 5}
labels_decoded = {y: x for x, y in labels_encoded.items()}

num_labels = len(labels_encoded)
