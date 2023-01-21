from variables import save_variables, read_variable
from preprocessing_text import *
from preprocessing_for_models import *
from eda import *
from modeling import *


def main():
    df1 = read_csv('activities_unlabeled.csv',
                   usecols=['File Name', 'Label'],
                   namecols=['filename', 'label'],
                   remove_nan='filename',
                   ignore_dash=True)

    htmls = read_htmls(df1, 'filename')

    toi_articles = read_articles(htmls)

    df_text1 = create_df_from_articles(df1, toi_articles)

    save_variables(variables={'df_text1': df_text1})

    df2 = read_csv('activities_labeled13.csv',
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


if '__name__' == '__main__':
    main()
