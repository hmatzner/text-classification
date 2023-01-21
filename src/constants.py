import os
import re
import spacy
from nltk.stem.porter import *
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')
nltk.download('stopwords')

MAIN_FOLDER = '/Users/hernanmatzner/text_classification/'
DATA_FOLDER = MAIN_FOLDER + 'data/'
HTML_FOLDER = MAIN_FOLDER + 'html_files_Nov-24-2022/'
VARIABLES_FOLDER = MAIN_FOLDER + 'saved_variables/'

URL = 'url'
TEXT = 'text'
LEMMATIZED = 'cleaned_lemmatized_text'
TARGET = 'label'

nlp = spacy.load('en_core_web_sm')

REPLACE_BY_SPACE_RE = re.compile(r'[/(){}\[\]|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9A-Za-z #+_]')
STOPWORDS = set(stopwords.words('english'))

labels_encoded = {'Article': 0, 'Blog': 1, 'Event': 2, 'Webinar': 3, 'PR': 4, 'MISC': 5}
labels_decoded = {y: x for x, y in labels_encoded.items()}

os.chdir(MAIN_FOLDER)