# python imports
import re
import collections

# third-party imports
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


'''
Find entities in text, replace strings with tags and extract tags:
    Donald Trump --> Donald_Trump
    [Donald Trump, PERSON]
'''


def utils_ner_text(txt, ner, lst_tag_filter=None, grams_join="_"):
    # apply model
    entities = ner(txt).ents

    # tag text
    tagged_txt = txt
    for tag in entities:
        if (lst_tag_filter is None) or (tag.label_ in lst_tag_filter):
            try:
                tagged_txt = re.sub(tag.text, grams_join.join(tag.text.split()), tagged_txt)
            except Exception:
                continue

    # extract tags list
    if lst_tag_filter is None:
        lst_tags = [(tag.text, tag.label_) for tag in entities]
    else:
        lst_tags = [(word.text, word.label_) for word in entities if word.label_ in lst_tag_filter]

    return tagged_txt, lst_tags


'''
Counts the elements in a list.
:parameter
    :param lst: list
    :param top: num - number of top elements to return
:return
    lst_top - list with top elements
'''


def utils_lst_count(lst, top=None):
    dic_counter = collections.Counter()
    for x in lst:
        dic_counter[x] += 1
    dic_counter = collections.OrderedDict(sorted(dic_counter.items(), key=lambda x: x[1], reverse=True))
    lst_top = [{key: value} for key, value in dic_counter.items()]
    if top is not None:
        lst_top = lst_top[:top]
    return lst_top


'''
Creates columns
    :param lst_dics_tuples: [{('Texas','GPE'):1}, {('Trump','PERSON'):3}]
    :param tag: string - 'PERSON'
:return
    int
'''


def utils_ner_features(lst_dics_tuples, tag):
    if len(lst_dics_tuples) > 0:
        tag_type = []
        for dic_tuples in lst_dics_tuples:
            for tuple in dic_tuples:
                type, n = tuple[1], dic_tuples[tuple]
                tag_type = tag_type + [type]*n
                dic_counter = collections.Counter()
                for x in tag_type:
                    dic_counter[x] += 1
        return dic_counter[tag]
    else:
        return 0


'''
Compute frequency of spacy tags.
'''


def tags_freq(tags, top, label, figsize=(10, 5)):
    tags_list = tags.sum()
    map_lst = list(map(lambda x: list(x.keys())[0], tags_list))
    dtf_tags = pd.DataFrame(map_lst, columns=['tag', 'type'])
    dtf_tags['count'] = 1
    dtf_tags = dtf_tags.groupby(['type', 'tag']).count().reset_index().sort_values('count', ascending=False)
    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle(f'Top frequent tags for {label.upper()} label', fontsize=12)
    sns.barplot(x='count', y='tag', hue='type', data=dtf_tags.iloc[:top, :], dodge=False, ax=ax)
    ax.set(ylabel=None)
    ax.grid(axis='x')
    plt.show()


'''
Apply spacy NER model and add tag features.
:parameter
    :param dtf: dataframe - dtf with a text column
    :param column: string - name of column containing text
    :param ner: spacy object - "en_core_web_lg", "en_core_web_sm", "xx_ent_wiki_sm"
    :param lst_tag_filter: list - ["ORG","PERSON","NORP","GPE","EVENT", ...]. If None takes all
    :param grams_join: string - "_", " ", or more (ex. "new york" --> "new_york")
    :param create_features: bool - create columns with category features
:return
    dtf
'''


def add_ner_spacy(data, column, ner=None, lst_tag_filter=None, grams_join='_', create_features=True):

    dtf = data.copy()

    # tag text and exctract tags
    dtf[[column + '_tagged', 'tags']] = dtf[[column]].apply(
        lambda x: utils_ner_text(x.iloc[0], ner, lst_tag_filter, grams_join), axis=1, result_type='expand')

    # put all tags in a column
    dtf['tags'] = dtf['tags'].apply(lambda x: utils_lst_count(x, top=None))

    # extract features
    if create_features:
        # features set
        tags_set = []
        for lst in dtf['tags'].tolist():
            for dic in lst:
                for k in dic.keys():
                    tags_set.append(k[1])
        tags_set = list(set(tags_set))
        # create columns
        for feature in tags_set:
            dtf['tags_' + feature] = dtf['tags'].apply(lambda x: utils_ner_features(x, feature))

    return dtf
