# python imports
import re
import contractions

# third-party imports
import pandas as pd

import nltk

import matplotlib.pyplot as plt
import seaborn as sns


'''
Preprocess a string.
:parameter
    :param txt: string - name of column containing text
    :param lst_regex: list - list of regex to remove
    :param punkt: bool - if True removes punctuations and characters
    :param lower: bool - if True convert lowercase
    :param slang: bool - if True fix slang into normal words
    :param lst_stopwords: list - list of stopwords to remove
    :param stemm: bool - whether stemming is to be applied
    :param lemm: bool - whether lemmitisation is to be applied
:return
    cleaned text
'''


def utils_preprocess_text(txt, lst_regex=None, punkt=True, lower=True, slang=True,
                          lst_stopwords=None, stemm=False, lemm=True):
    # Regex (in case, before cleaning)
    if lst_regex is not None:
        for regex in lst_regex:
            txt = re.sub(regex, '', txt)

    # separate sentences with '. '
    txt = re.sub(r'\.(?=[^ \W\d])', '. ', str(txt))
    # remove punctuations and characters
    txt = re.sub(r'[^\w\s]', '', txt) if punkt is True else txt
    # strip
    txt = ' '.join([word.strip() for word in txt.split()])
    # lowercase
    txt = txt.lower() if lower is True else txt
    # slang
    txt = contractions.fix(txt) if slang is True else txt

    # Tokenize (convert from string to list)
    lst_txt = txt.split()

    # Stemming (remove -ing, -ly, ...)
    if stemm is True:
        ps = nltk.stem.porter.PorterStemmer()
        lst_txt = [ps.stem(word) for word in lst_txt]

    # Lemmatization (convert the word into root word)
    if lemm is True:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        lst_txt = [lem.lemmatize(word) for word in lst_txt]

    # Stopwords
    if lst_stopwords is not None:
        lst_txt = [word for word in lst_txt if word not in lst_stopwords]

    # Back to string
    txt = ' '.join(lst_txt)
    return txt


'''
Adds a column of preprocessed text.
:parameter
    :param dtf: dataframe - dtf with a text column
    :param column: string - name of column containing text
:return
    : input dataframe with two new columns
'''


def add_preprocessed_text(data, column, lst_regex=None, punkt=False, lower=False, slang=False,
                          lst_stopwords=None, stemm=False, lemm=False, remove_na=True):
    dtf = data.copy()

    # apply preprocess
    dtf = dtf[pd.notnull(dtf[column])]
    dtf[column+'_clean'] = dtf[column].apply(
        lambda x: utils_preprocess_text(x, lst_regex, punkt, lower, slang, lst_stopwords, stemm, lemm))

    # residuals
    dtf['check'] = dtf[column + '_clean'].apply(lambda x: len(x))
    if dtf['check'].min() == 0:
        print(dtf[[column, column + '_clean']][dtf['check'] == 0].head())
        if remove_na is True:
            dtf = dtf[dtf['check'] > 0]

    return dtf.drop('check', axis=1)


'''
Compute n-grams frequency with nltk tokenizer.
:parameter
    :param corpus: list - dtf["text"]
    :param ngrams: int or list - 1 for unigrams, 2 for bigrams, [1,2] for both
    :param top: num - plot the top frequent words
:return
    dtf_count: dtf with word frequency
'''


def word_freq(corpus, label, ngrams=[1, 2, 3], top=10, figsize=(10, 7)):
    lst_tokens = nltk.tokenize.word_tokenize(corpus.str.cat(sep=' '))
    ngrams = [ngrams] if type(ngrams) is int else ngrams

    # calculate
    dataframes = list()

    for n in ngrams:
        dic_words_freq = nltk.FreqDist(nltk.ngrams(lst_tokens, n))
        dtf_n = pd.DataFrame(dic_words_freq.most_common(), columns=['word', 'freq'])
        dtf_n['ngrams'] = n
        dataframes.append(dtf_n)

    dtf_freq = pd.concat(dataframes)
    dtf_freq['word'] = dtf_freq['word'].apply(lambda x: ' '.join(string for string in x))
    dtf_freq = dtf_freq.sort_values(['ngrams', 'freq'], ascending=[True, False])

    # plot
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(x='freq', y='word', hue='ngrams', dodge=False, ax=ax,
                data=dtf_freq.groupby('ngrams')[['ngrams', 'freq', 'word']].head(top))
    ax.set(xlabel=None, ylabel=None, title=f'Most frequent words for {label.upper()} label')
    ax.grid(axis='x')
    plt.show()
