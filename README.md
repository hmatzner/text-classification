# Text Classification Project

This project involves the classification of text data that consists of 
URLs and HTML files. The goal is to take as input either a URL or an 
HTML file and predict the subject matter of the text. The categories 
or labels that the model will predict are Webinar, Event, Press 
release, Article, Blog, and MISC. MISC is a catch-all category for 
texts that do not belong to any of the other five categories.

### Getting Started
The code can be executed from both .ipynb or .py versions, which are 
located in the main directory. In order to work, the two csv files must 
be located in the same directory, while the HTML files to read will be
located in a folder named `html_files_Nov-24-2022`.
For NDA reasons, the files used in this real-world project are not public.
Variables and models will be saved in their respective directories, for
what we will need to execute in the main directory the commands:
```
! mkdir models
! mkdir saved_variables
'''

### Prerequisites

To run the code in this file, you will need to install the following 
libraries:

- urllib3 (version 1.26.6)
- pandas (version 1.4.3)
- numpy (version 1.21.2)
- matplotlib (version 3.4.3)
- seaborn (version 0.11.2)
- spacy (version 3.4.2)
- tqdm (version 4.64.0)
- newspaper3k (version 0.2.8)
- scikit-learn (version 1.1.1)
- datasets (version 2.8.0)
- transformers (version 4.25.1)
- tensorflow (version 2.11.0)
- nltk (version 3.7)

To install the required libraries and dependencies using pip, you can 
use the following command:

`pip install -r requirements.txt`
