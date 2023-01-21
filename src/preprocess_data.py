from variables import save_variables, read_variable
import constants
from constants import VARIABLES_FOLDER, DATA_FOLDER, HTML_FOLDER, TARGET, TEXT, URL, LEMMATIZED

import os
import pandas as pd
import numpy as np
from typing import List, Tuple, Any
from tqdm import tqdm
import newspaper
from newspaper import Article

if not os.path.isdir(VARIABLES_FOLDER):
    os.makedirs(VARIABLES_FOLDER)

if not os.path.isdir(HTML_FOLDER):
    raise Exception('HTML folder with relevant files should be already created and located in the main folder.')


def read_csv(csv_path: str, usecols: List[str] = None, namecols: List[str] = None, remove_nan: str = None,
             ignore_dash: bool = False) -> pd.DataFrame:
    """
    Creates a DataFrame from a CSV file.

    Parameters:
    - csv_path: the path to the CSV file.
    - usecols: a list of column names to use from the CSV file. If not provided, all columns will be used.
    - namecols: a list of new names for the columns from the CSV file. If not provided, the original column names will
    be used.
    - remove_nan: the name of the column from where rows with missing values will be removed. It must be from namecols
    (or from usecols, if namecols was not provided). If not provided, all rows will be kept.
    - ignore_dash: a flag indicating whether to ignore rows with a dash (`-`) in the target column. If not provided,
    rows with a dash in the target column will be included in the data.

    Returns:
    - df: a pandas DataFrame object.
    """

    df = pd.read_csv(csv_path, usecols=usecols)

    if namecols:
        assert len(usecols) == len(namecols)
        rename_columns = {usecols[i]: namecols[i] for i in range(len(usecols))}
        df.rename(columns=rename_columns, inplace=True)

    if remove_nan:
        df = df[~df[remove_nan].isna()]

    if ignore_dash:
        df = df[df[TARGET] != '-']

    # All sections of blogs labeled as 'MISC/Blog?' become part of 'MISC'.
    df.loc[df[TARGET] == 'MISC/Blog?', TARGET] = 'MISC'

    df.reset_index(drop=True, inplace=True)

    return df


def read_htmls(df: pd.DataFrame, column: str) -> List[str]:
    """
    Reads HTML files from a pandas DataFrame.

    Parameters:
    - df: a DataFrame with a column containing the names of the HTML files.
    - column: the name of the column in the DataFrame where the filenames are stored.

    Returns:
    - htmls: a list of strings, one for each HTML file.
    """

    filenames = df[column].values
    htmls = list()

    for i, filename in enumerate(tqdm(filenames)):
        try:
            with open(f'{HTML_FOLDER}{filename}') as f:
                html = f.read()
                htmls.append(html)
        except FileNotFoundError:
            print(f'File {i} not found: "{filename}"')

    return htmls


def read_articles(htmls: List[str]) -> List[newspaper.article.Article]:
    """
    Reads articles from a list of HTML strings.

    Parameters:
    - htmls: a list of HTML strings representing articles.

    Returns:
    - toi_articles: a list of newspaper.article.Article objects, one for each HTML string.
    """

    toi_articles = list()

    for html in tqdm(htmls):
        toi_article = Article(url=' ', language='en')
        toi_article.set_html(html)
        toi_article.parse()
        toi_article.nlp()
        toi_articles.append(toi_article)

    return toi_articles


def create_df_from_articles(df: pd.DataFrame, toi_articles: List[newspaper.article.Article]) -> pd.DataFrame:
    """
    Creates a DataFrame with article text and labels from a list of articles.

    Parameters:
    - df: a DataFrame with a TARGET column containing the labels for the articles.
    - toi_articles: a list of newspaper.article.Article objects.

    Returns: a DataFrame with two columns: TEXT, containing the concatenated title and text of the articles, and TARGET,
    containing the labels for the articles.
    """

    summaries = [(toi_article.title + '. ' + toi_article.text).replace('\n', ' ') for toi_article in toi_articles]
    y = df[TARGET].tolist()
    assert len(y) == len(summaries)

    df_text1 = pd.DataFrame({TEXT: summaries, TARGET: y})

    return df_text1


def read_or_create_variables(variable_names: List[str]) -> List[Any]:
    """
    Reads or creates variables with the given names. If a variable exists, it is read and returned.
    If a variable with a given name does not exist, it is created and an empty list is returned.

    Parameters:
    - variable_names: list of strings representing the names of the variables to be read and/or created.

    Returns:
    - variables: list of the read and/or created variables.
    """

    variables = list()

    for variable_name in variable_names:
        try:
            variable = read_variable(variable_name)
            print(f'SUCCESS: variable {variable_name} was read, it contains {len(variable)} elements.')
        except FileNotFoundError:
            variable = list()
            print(f'Variable {variable_name} was just created and contains {len(variable)} elements.')

        variables.append(variable)

    return variables


def create_new_urls(df: pd.DataFrame, column: str, urls_old: List[str]) -> pd.Series:
    """
    Creates the urls to read based on the difference between all URLs and the ones already read.

    Parameters:
    - df: a DataFrame with a column containing the URLs.
    - column: the name of the column in the DataFrame where the URLs are stored.
    - urls_old: list of strings of the URLs that have already been read.

    Returns:
    - urls_new: pandas Series object with strings of the URLs to read.
    """

    urls = df[column]
    mask = np.isin(urls, urls_old)
    urls_new = urls[~mask]
    print(f'There are {len(urls_new)} new URLs to read.')

    return urls_new


def read_texts_from_urls(urls_new: pd.Series, urls_old: List[str]) -> Tuple[List[str], List[int], List[int]]:
    """
    Reads the relevant text from each URL.

    Parameters:
    - urls_new: pandas Series object with strings of the URLs to read.
    - urls_old: list of strings of the URLs that have already been read.

    Returns a tuple of three elements:
    - texts_new: list of strings of the texts read from the URLs.
    - indexes_new: list of integers representing the indexes of the URLs that were read successfully.
    - idx_label_to_remove: list of integers representing the indexes, starting from 0, that threw errors and should be
    deleted.
    """

    texts_new = list()
    indexes_new = list()

    idx_label_to_remove = list()
    idx_label = 0

    for i, url in enumerate(urls_new, len(urls_old)):
        try:
            toi_article = Article(url=url, language='en')
            toi_article.download()
            toi_article.parse()
            toi_article.nlp()
            text = (toi_article.title + '. ' + toi_article.text).replace('\n', ' ')
            texts_new.append(text)
            indexes_new.append(i)
            print(f'{i}: url "{url}" read successfully.')
        except Exception:
            idx_label_to_remove.append(idx_label)
            print(f'{i}: ERROR: url "{url}" was not read successfully.')

        idx_label += 1

    if idx_label_to_remove:
        print(f'\nWhen reading the new URLs, {len(urls_new) - len(indexes_new)} '
              f'of them threw an error and could not be read.')
    elif texts_new:
        print('All URLs were successfully read.')
    else:
        print('No new URL was read.')

    return texts_new, indexes_new, idx_label_to_remove


def create_new_labels(df: pd.DataFrame, urls_to_read: pd.Series, idx_label_to_remove: List[int]) -> List[str]:
    """
    Creates a list of labels for the given URLs and removes the labels corresponding to the URLs that threw errors.

    Parameters:
    - df: pandas DataFrame containing the labels.
    - urls_to_read: pandas Series object with strings of the URLs for which the labels should be created.
    - idx_label_to_remove: list of integers representing the indexes of the URLs that threw errors and should have their
    labels removed.

    Returns:
    - labels_new: list of labels according to the new URLs, discarding the ones that threw errors.
    """

    labels_new = df.loc[urls_to_read.index, TARGET]
    labels_new.reset_index(drop=True, inplace=True)
    labels_new = labels_new.drop(idx_label_to_remove).tolist()

    return labels_new


def update_variables(old_variables: List[Any], new_variables: List[Any]) -> List[Any]:
    """
    Updates the old variables with the new variables.

    Parameters:
    - old_variables: list of variables that should be updated.
    - new_variables: list of variables used to update the old variables, should be of same length and order as
    `old_variables`.

    Returns:
    - updated_variables: list of variables that are the result of updating the old variables with the new variables.
    """

    assert len(old_variables) == len(new_variables)

    updated_variables = list()
    zipped_variables = list(zip(old_variables, new_variables))

    for old_var, new_var in zipped_variables:
        updated_variable = old_var + new_var
        updated_variables.append(updated_variable)

    return updated_variables


def create_df_from_lists(labels: List[str], indexes: List[int], texts: List[str], urls: List[str]) -> pd.DataFrame:
    """
    Creates a pandas DataFrame with columns for URLs, texts, and labels.

    Parameters:
    - labels: list of strings representing the labels of the articles.
    - indexes: list of integers representing the indexes of the articles.
    - texts: list of strings representing the texts of the articles.
    - urls: list of strings representing the URLs of the articles.

    Returns:
    - df_text2: pandas DataFrame with columns for URLs, texts, and labels.
    """

    df_text2 = pd.DataFrame({
        URL: pd.Series(urls).loc[indexes],
        TEXT: texts,
        TARGET: labels,
    }).reset_index(drop=True)

    return df_text2


def remove_duplicates(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Removes duplicate rows from a DataFrame based on a specific column.

    Parameters:
    - df: a DataFrame with a column containing values to be checked for duplicates.
    - column: the name of the column in the DataFrame where the values are stored.

    Returns:
    - df: DataFrame with duplicate rows removed. If there were no duplicates, the original DataFrame is returned.
    """

    if df[column].duplicated().any():
        original_amount = df.shape[0]
        df = df[~df[column].duplicated()]
        print(f"{original_amount - df.shape[0]} rows had duplicate values in the dataframe's column '{column}' "
              f"and were deleted.")
        assert not df[column].duplicated().any()

    else:
        print(f"There are no duplicate values in the dataframe's column '{column}'.")

    return df


def remove_rows(df: pd.DataFrame, with_errors: bool = False,
                irrelevant: bool = False, below_threshold: int = None) -> pd.DataFrame:
    """
    Removes rows from a pandas DataFrame that meet certain conditions.

    Parameters:
    - df: pandas DataFrame from which rows should be removed.
    - with_errors: boolean flag indicating whether rows with URLs' texts that throw errors should be removed. If set to
    True, rows with texts that have less than 100 words and contain both the strings " 404 " and " error " will be
    removed.
    - irrelevant: boolean flag indicating whether rows with irrelevant labels should be removed. If set to True, rows
    with labels that are not included in the dictionary `labels_encoded` will be removed.
    - below_threshold: integer representing the minimum number of words that the text of a URL must contain for it to be
    included in the final DataFrame. Rows with texts containing fewer words than this threshold will be removed.

    Returns:
    - df: pandas DataFrame with rows that met a condition removed.
    """

    if with_errors:
        words = df[TEXT].str.split().str.len()
        condition1 = words < 100  # empirical threshold
        condition2 = df[TEXT].str.contains(' 404 ')
        condition3 = df[TEXT].str.contains(' error ')
        df = df[~(condition1 & condition2 & condition3)]

    if irrelevant:
        df = df[df[TARGET].isin(constants.labels_encoded)]
        df.reset_index(drop=True, inplace=True)

    # If the words that the text a URL contains is below this threshold, the row is discarded
    if below_threshold:
        words = df[TEXT].str.split().str.len()
        df = df[words >= below_threshold]

    return df


def text_preprocessing(text: str, lemmatize: bool = False, clean: bool = False) -> str:
    """
    Preprocess text by applying lemmatization and cleaning operations.

    Parameters:
    - text: a string of the text to be preprocessed.
    - lemmatize: a flag indicating whether to apply lemmatization to the text. Default is False.
    - clean: a flag indicating whether to clean the text. Default is False.

    Returns:
    - text: a string of the preprocessed text.
    """

    if lemmatize:
        text = ' '.join(token.lemma_ for token in constants.nlp(text))

    if clean:
        text = constants.REPLACE_BY_SPACE_RE.sub(' ', text)
        text = constants.BAD_SYMBOLS_RE.sub('', text)
        text = ' '.join(word for word in text.split() if word not in constants.STOPWORDS)

    return text


def main():
    """
    Main function of the module that reads in two CSV files, processes the data,
    and saves the resulting dataframes to be used later by any model.

    Parameters:
    - None

    Returns:
    - None
    """

    df1 = read_csv(f'{DATA_FOLDER}activities_unlabeled.csv',
                   usecols=['File Name', 'Label'],
                   namecols=['filename', 'label'],
                   remove_nan='filename',
                   ignore_dash=True)

    htmls = read_htmls(df1, 'filename')

    toi_articles = read_articles(htmls)

    df_text1 = create_df_from_articles(df1, toi_articles)

    save_variables(variables={'df_text1': df_text1})

    df2 = read_csv(f'{DATA_FOLDER}activities_labeled13.csv',
                   usecols=['url', 'true_label'],
                   namecols=['url', 'label'],
                   remove_nan='label')

    labels_old, indexes_old, texts_old, urls_old = read_or_create_variables(
        ['labels_old', 'indexes_old', 'texts_old', 'urls_old'])

    urls_new = create_new_urls(df2, 'url', urls_old)

    texts_new, indexes_new, idx_label_to_remove = read_texts_from_urls(urls_new, urls_old)

    labels_new = create_new_labels(df2, urls_new, idx_label_to_remove)

    labels, indexes, texts, urls = update_variables(old_variables=[labels_old, indexes_old, texts_old, urls_old],
                                                    new_variables=[labels_new, indexes_new, texts_new,
                                                                   urls_new.tolist()])

    assert len(texts) == len(pd.Series(urls).loc[indexes]) == len(indexes) == len(labels)

    df_text2 = create_df_from_lists(labels, indexes, texts, urls)

    save_variables(variables={'labels_old': labels,
                              'indexes_old': indexes,
                              'texts_old': texts,
                              'urls_old': urls,
                              'df_text2': df_text2})

    df_text2 = remove_duplicates(df_text2, URL)

    df_text = pd.concat([df_text1, df_text2]).reset_index(drop=True)

    df_text = remove_duplicates(df_text, TEXT)

    df_text = remove_rows(df_text, with_errors=True, irrelevant=True, below_threshold=51)

    df_text[LEMMATIZED] = df_text[TEXT].apply(lambda x: text_preprocessing(x, lemmatize=True, clean=True))

    save_variables({'df_text': df_text})


if __name__ == '__main__':
    main()
