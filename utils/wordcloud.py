# third-party imports
import wordcloud

import matplotlib.pyplot as plt


'''
Plots a wordcloud from a list of Docs or from a dictionary
:parameter
    :param corpus: list - dtf["text"]
'''


def plot_wordcloud(corpus, label, max_words=150, max_font_size=35, figsize=(10, 10)):
    wc = wordcloud.WordCloud(background_color='black', max_words=max_words, max_font_size=max_font_size)
    wc = wc.generate(str(corpus))

    fig, ax = plt.subplots(num=1, figsize=figsize)
    ax.set(xlabel=None, ylabel=None, title=f'Wordcloud for {label.upper()} label')
    plt.axis('off')
    plt.imshow(wc, cmap=None)
    plt.show()
