# Text Classification Project

This project involves the classification of text data that consists of 
URLs and HTML files. The goal is to take as input either a URL or an 
HTML file and predict the subject matter of the text. The categories 
or labels that the model will predict are Webinar, Event, Press 
release, Article, Blog, and MISC. MISC is a catch-all category for 
texts that do not belong to any of the other five categories.

### Getting Started
In order to work, the two csv files must be located in the same directory, 
while the HTML files to read will be located in a folder named `html_files_Nov-24-2022`.
For NDA reasons, the files used in this real-world project are not public.
Variables and models will be saved in their respective directories, for
what we will need to execute in the main directory the commands:
```
! mkdir models
! mkdir saved_variables
```

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

```
pip install -r requirements.txt
```

### Code Structure

In this project, there are nine python files located in `src`, 
each with a specific purpose:

`constants.py`: This file contains constants that are imported and used by 
different files.
`variables.py`: This file includes functions that simplify the process of 
reading and saving variables.
`preprocessing.py`: This file includes functions for the preprocessing of 
all three approaches.
`eda.py`: This file includes functions to perform exploratory data 
analysis 
in all three approaches.
`modeling.py`: This file includes functions for the modeling of all three 
approaches.
`bow_ml.py`: This file executes the first approach, which consists of a 
Bag 
of Words and TF-IDF + Machine Learning model.
`distilbert_ml.py`: This file executes the second approach, which consists 
of fine-tuning a pre-trained DistilBERT model with a Machine Learning 
model.
`distilbert_dl.py`: This file executes the third approach, which consists 
of 
fine-tuning a pre-trained DistilBERT model with a Deep Learning model.

It is important to note that the data used in this project is 
confidential, and therefore not publicly available. 
