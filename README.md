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
Each '___' is of type string and is the name of one of the true labels.

### Getting Started
In order to work, the two CSV files must be located in the same directory, 
while the HTML files to read will be located in a folder named `htmls`.
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
- `preprocessing.py`: This file includes functions for the preprocessing of
all three approaches.
- `eda.py`: This file includes functions to perform exploratory data
analysis
in all three approaches.
- `modeling.py`: This file includes functions for the modeling of all three
approaches.

#### The execution files are:
- `preprocess_data.py`: This file executes the general preprocessing step by
creating and saving the dataframe `df_text` that will be needed in all three
approaches. This is the file that must be executed first.
- `bow_ml.py`: This file executes the first approach, which consists of
Bag of Words and TF-IDF + Machine Learning model.
- `distilbert_ml.py`: This file executes the second approach, which consists
of fine-tuning the pre-trained DistilBERT embeddings with a Machine Learning
model.
- `distilbert_dl.py`: This file executes the third approach, which consists
of fine-tuning the pre-trained DistilBERT embeddings  with a Deep Learning model.
