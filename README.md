# Text Classification Project

This project involves the classification of text data that consists of 
URLs and HTML files with their true label, each type in a different CSV file. 
The goal is to take as input either one of the two
and predict the subject matter of the text. The categories 
or labels that the model will predict are not disclosed,
and are located in the `labels.py` file, which contains a unique constant:
```
labels_encoded = {'___': 0, '___': 1, '___': 2, '___': 3, '___': 4, '___': 5}
```
Each `'___'` is of type string and is the name of one of the true labels.

We will be implementing three different approaches (models) to perform this
task and compare their results.

### Getting Started
In order to work, the two CSV files must be located in `data/`, 
while the HTML files to read will be located in `htmls/`.
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

### Code structure and execution

In this project, there are 10 python files located in `src`, 
having six of them as main purpose being used by other modules and four
of them to perform the execution.

#### The files to serve the execution files are:
- `labels.py`: This file is hidden and contains a unique variable with the
true labels encoded.
- `constants.py`: This file contains constants that are imported and used by
different files.
- `variables.py`: This file includes functions that simplify the process of
reading and saving variables.
- `preprocessing.py`: This file includes functions for all the particular 
preprocessing steps any of the three approaches need.
- `eda.py`: This file includes functions to perform exploratory data
analysis with all three approaches.
- `modeling.py`: This file includes functions for the modeling part of all three
approaches.

#### The execution files are:
1. `preprocess_data.py`: This file executes
the general preprocessing steps by creating and saving a dataframe necessary to
perform any of the approaches.
2. `bow_ml.py`: This file executes the first approach, which consists of
Bag of Words and TF-IDF + Machine Learning model.
3. `distilbert_ml.py`: This file executes the second approach, which consists
of fine-tuning the pre-trained DistilBERT embeddings with a Machine Learning
model.
4. `distilbert_dl.py`: This file executes the third approach, which consists
of fine-tuning the pre-trained DistilBERT embeddings  with a Deep Learning model.

The only files to be run by you are the ones for execution.
The first execution file needs to be run first in order to run the other ones.
The other three files for execution can be then run independently.
