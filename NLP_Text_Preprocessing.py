

import numpy as np
import nltk
from nltk import sent_tokenize, word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer
import matplotlib.pyplot as plt


"""
Natural Language Processing (NLP) with Python - Tutorial


Refer-
https://towardsai.net/p/nlp/natural-language-processing-nlp-with-python-tutorial-for-beginners-1f54e610a1a0
https://wortschatz.uni-leipzig.de/en/download/English
"""


# Read in text file-
with open("eng_news_2020_10K-sentences.txt", encoding = "utf8") as text_file:
    text = text_file.read()

type(text), len(text)
# (str, 1221374)

# Tokenize text as sentences-
sentences = sent_tokenize(text)

type(sentences), len(sentences)
# (list, 9549)

print(f"There are {len(sentences)} sentences")
# There are 9549 sentences

# Print first two sentences-
sentences[0]
# '1\t“18 months ago, we expelled a boy at Nations for selling drugs in six schools.'

sentences[1]
# '2\t” 41 Nigeria Centre for Disease Control (NCDC) staff and 17 World Health Organisation (WHO) staff are deployed at the moment to support the Kano response.'


# Tokenize text as words-
words = word_tokenize(text)

type(words), len(words)
# (list, 236877)

# Print a few words-
for i in range(5):
    print(words[i])
'''
1
“
18
months
ago
'''


# Compute frequencies-
word_freq_dist = FreqDist(words)

# Print 5 most common words-
word_freq_dist.most_common(5)
# [('the', 10235), (',', 9624), ('.', 9201), ('to', 5577), ('of', 4986)]

# Visualize top-5 frequency plot-
word_freq_dist.plot(5)
plt.show()


# List to contain words without punctuations-
words_no_punc = []

# Remove punctuations-
for wrd in words:
    # Use 'isalpha()' method to separate punctuation marks from actual text-
    if wrd.isalpha():
        words_no_punc.append(wrd.lower())


len(words), len(words_no_punc)
# (236877, 191812)

print(f"number of punctuations removed = {len(words) - len(words_no_punc)}")
# number of punctuations removed = 45065

# Print first 5 words with no punctuations-
words_no_punc[:5]
# ['months', 'ago', 'we', 'expelled', 'a']

# Compute frequencies-
word_no_punc_freq_dist = FreqDist(words_no_punc)

# Print 5 most common words having no punctuations-
word_no_punc_freq_dist.most_common(5)
# [('the', 11724), ('to', 5634), ('and', 5052), ('of', 5006), ('a', 4396)]


# Get list of English language stopwords-
stopwords = stopwords.words("english")

len(stopwords)
# 179

stopwords[:10]
# ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're"]


# Remove stopwords:

# List to contain clean words-
clean_words = []

for wrd in words_no_punc:
    if wrd not in stopwords:
        clean_words.append(wrd)

len(words_no_punc), len(clean_words)
# (191812, 104656)

print(f"number of stopwords removed = {len(words_no_punc) - len(clean_words)}")
# number of stopwords removed = 87156


# Get final frequency distribution for clean words-
clean_words_freq_dist = FreqDist(clean_words)

# Get 10 most common clean words-
clean_words_freq_dist.most_common(10)
'''
[('said', 1041),
 ('also', 467),
 ('people', 442),
 ('new', 425),
 ('one', 420),
 ('would', 343),
 ('time', 286),
 ('like', 277),
 ('state', 254),
 ('first', 254)]
'''




"""
Stemming:
We use Stemming to normalize words. In English and many other languages, a single word
can take multiple forms depending upon context used. For instance, the verb “study” can
take many forms like “studies,” “studying,” “studied,” and others, depending on its context.
When we tokenize words, an interpreter considers these input words as different words
even though their underlying meaning is the same. Moreover, as we know that NLP is about
analyzing the meaning of content, to resolve this problem, we use stemming.

Stemming normalizes the word by truncating the word to its stem word. For example, the words
“studies,” “studied,” “studying” will be reduced to “studi,” making all these word forms to
refer to only one token. Notice that stemming may not give us a dictionary, grammatical word
for a particular set of words.
"""
# Initialize a Porter Stemmer-
porter_stem = PorterStemmer()

# Word-list for stemming-
word_list = ['Study', 'Studying', 'Studies', 'Studied']

for wrd in word_list:
    print(porter_stem.stem(wrd))
'''
studi
studi
studi
studi
'''


'''
SnowballStemmer:
SnowballStemmer generates the same output as porter stemmer, but it supports many more languages.

snowball = SnowballStemmer("english")
'''

print(f"Languages supported by SnowballStemmer are: {SnowballStemmer.languages}")
# Languages supported by SnowballStemmer are: ('arabic', 'danish', 'dutch', 'english', 'finnish', 'french', 'german', 'hungarian', 'italian', 'norwegian', 'porter', 'portuguese', 'romanian', 'russian', 'spanish', 'swedish')

# Refer to tutorial for other stemmers.




"""
Lemmatization:

Lemmatization tries to achieve a similar base “stem” for a word. However, what makes it
different is that it finds the dictionary word instead of truncating the original word.
Stemming does not consider the context of the word. That is why it generates results faster,
but it is less accurate than lemmatization.

If accuracy is not the project’s final goal, then stemming is an appropriate approach. If
higher accuracy is crucial and the project is not on a tight deadline, then the best option
is amortization (Lemmatization has a lower processing speed, compared to stemming).

Lemmatization takes into account Part Of Speech (POS) values. Also, lemmatization may generate
different outputs for different values of POS. We generally have four choices for POS:
"""
word_list
# ['Study', 'Studying', 'Studies', 'Studied']

for wrd in word_list:
    print(f"WordNet lemmatizer: {lemmatizer.lemmatize(wrd, pos = 'v')}")
'''
WordNet lemmatizer: Study
WordNet lemmatizer: Studying
WordNet lemmatizer: Studies
WordNet lemmatizer: Studied
'''

# The default value of PoS in lemmatization is a noun(n). In the following example, we can see that
# it’s generating dictionary words-
word_list = ['studies', 'leaves', 'decreases', 'plays']

for wrd in word_list:
    print(f"WordNet lemmatizer: {lemmatizer.lemmatize(wrd)}")
'''
WordNet lemmatizer: study
WordNet lemmatizer: leaf
WordNet lemmatizer: decrease
WordNet lemmatizer: play
'''


# Get lemmatized words for the cleaned vocabulary-
words_lemmatized = []

for wrd in clean_words:
    words_lemmatized.append(lemmatizer.lemmatize(wrd))

len(words_lemmatized), len(clean_words)
# (104656, 104656)

len(set(words_lemmatized)), len(set(clean_words))
# (18623, 20773)

# Get frequency of lemmatized and cleaned words-
words_lemmatized_freq_dist = FreqDist(clean_words)

# Get 15 most common words-
words_lemmatized_freq_dist.most_common(15)
'''
[('said', 1041),
 ('also', 467),
 ('people', 442),
 ('new', 425),
 ('one', 420),
 ('would', 343),
 ('time', 286),
 ('like', 277),
 ('state', 254),
 ('first', 254),
 ('two', 250),
 ('year', 230),
 ('health', 228),
 ('last', 227),
 ('could', 208)]
'''


"""
# BAD IDEA: Using set() will remove frequency information!!

# Get unique lemmatized and cleaned words as a Python3 list-
words_lemmatized = list(set(words_lemmatized))

# Sanity check-
len(words_lemmatized)
# 18623

# Compute frequency distribution-
words_lemmatized_freq_dist = FreqDist(words_lemmatized)

# Get 15 most common words-
words_lemmatized_freq_dist.most_common(15)
'''
[('trooper', 1),
 ('rotate', 1),
 ('lastella', 1),
 ('profire', 1),
 ('pedestal', 1),
 ('lite', 1),
 ('accurately', 1),
 ('ole', 1),
 ('measuring', 1),
 ('amnesty', 1),
 ('assumption', 1),
 ('whiner', 1),
 ('thunderbrand', 1),
 ('footprint', 1),
 ('acutely', 1)]
'''
"""

