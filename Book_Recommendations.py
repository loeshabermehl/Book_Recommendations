import nltk
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize, sent_tokenize
from nltk import ngrams
from itertools import groupby, chain
import re
import string
import scipy
import emoji
from gutenberg.acquire import load_etext
from gutenberg.cleanup import strip_headers

# Download the necessary sets from the Natural Language Toolkit, you can comment the next lines if already present.
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('gutenberg')

stop_words = set(stopwords.words('english'))
book_id = 0

function_words = ['a',
                  'between',
                  'in',
                  'nor',
                  'some',
                  'upon',
                  'about',
                  'both',
                  'including',
                  'nothing',
                  'somebody',
                  'us',
                  'above',
                  'but',
                  'inside',
                  'of',
                  'someone',
                  'used',
                  'after',
                  'by',
                  'into',
                  'off',
                  'something',
                  'via',
                  'all',
                  'can',
                  'is',
                  'on',
                  'such',
                  'we',
                  'although',
                  'cos',
                  'it',
                  'once',
                  'than',
                  'what',
                  'am',
                  'do',
                  'its',
                  'one',
                  'that',
                  'whatever',
                  'among',
                  'down',
                  'latter',
                  'onto',
                  'the',
                  'when',
                  'an',
                  'each',
                  'less',
                  'opposite',
                  'their',
                  'where',
                  'and',
                  'either',
                  'like',
                  'or',
                  'them',
                  'whether',
                  'another',
                  'enough',
                  'little',
                  'our',
                  'these',
                  'which',
                  'any',
                  'every',
                  'lots',
                  'outside',
                  'they',
                  'while',
                  'anybody',
                  'everybody',
                  'many',
                  'over',
                  'this',
                  'who',
                  'anyone',
                  'everyone',
                  'me',
                  'own',
                  'those',
                  'whoever',
                  'anything',
                  'everything',
                  'more',
                  'past',
                  'though',
                  'whom',
                  'are',
                  'few',
                  'most',
                  'per',
                  'through',
                  'whose',
                  'around',
                  'following',
                  'much',
                  'plenty',
                  'till',
                  'will',
                  'as',
                  'for',
                  'must',
                  'plus',
                  'to',
                  'with',
                  'at',
                  'from',
                  'my',
                  'regarding',
                  'toward',
                  'within',
                  'be',
                  'have',
                  'near',
                  'same',
                  'towards',
                  'without',
                  'because',
                  'he',
                  'need',
                  'several',
                  'under',
                  'worth',
                  'before',
                  'her',
                  'neither',
                  'she',
                  'unless',
                  'would',
                  'behind',
                  'him',
                  'no',
                  'should',
                  'unlike',
                  'yes',
                  'below',
                  'i',
                  'nobody',
                  'since',
                  'until',
                  'you',
                  'beside',
                  'if',
                  'none',
                  'so',
                  'up',
                  'your']
pos_tags = ['CC',
            'CD',
            'DT',
            'EX',
            'FW',
            'IN',
            'JJ',
            'JJR',
            'JJS',
            'LS',
            'MD',
            'NN',
            'NNS',
            'NNP',
            'NNPS',
            'PDT',
            'POS',
            'PRP',
            'PRP$',
            'RB',
            'RBR',
            'RBS',
            'RP',
            'SYM',
            'TO',
            'UH',
            'VB',
            'VBD',
            'VBG',
            'VBN',
            'VBP',
            'VBZ',
            'WDT',
            'WP',
            'WP$',
            'WRB']
IDs = [1342,
       46,
       98,
       84,
       1661,
       219,
       2701,
       11,
       1080,
       345,
       43,
       844,
       76,
       6130,
       74,
       30254,
       58585,
       2554,
       2591,
       5200,
       2542,
       2600,
       16328,
       174,
       1400,
       1184,
       1260,
       1497,
       4300,
       205,
       160,
       28054,
       25344,
       135,
       120,
       1322,
       1952,
       158,
       16,
       1232,
       2814,
       2500,
       408,
       768,
       27827,
       3207,
       55,
       863,
       8800,
       244,
       30360,
       514,
       2680,
       2852,
       4363,
       10,
       100,
       20203,
       3600,
       36,
       45,
       1998,
       161,
       35,
       996,
       37423,
       730,
       829,
       28520,
       17135,
       25717,
       1404,
       203,
       61,
       1399,
       786,
       236,
       1727,
       3825,
       521,
       113,
       19942,
       23,
       19337,
       1934,
       2097,
       57426,
       41,
       766,
       34901,
       375,
       1155,
       209,
       2148,
       1250,
       972,
       600,
       25305,
       20228,
       58685]


# Extract features from a given text
def extract_features(text, common_ngrams):
    bag_of_words = wordpunct_tokenize(text)
    bag_of_sentences = sent_tokenize(text)

    features = []

    # Feature 1: count the total number of characters
    features.append(len(text))

    # Feature 2: count the total number of words
    features.append(len(bag_of_words))

    # Feature 3: count the number of words, excluded the stopwords
    features.append(len([x for x in bag_of_words if x.lower() not in stop_words]))

    # Feature 4: count the total number of sentences
    features.append(len(bag_of_sentences))

    # Feature 5: count the frequency of words without vowels
    features.append(len([x for x in bag_of_words if re.search('^[^aeiouAEIOU]+$', x)]))

    # Feature 6: count the total number of alphabets
    features.append(len([x for x in text if x.isalpha()]))

    # Feature 7: count the total number of punctuations
    features.append(len([x for x in bag_of_words if x in string.punctuation]))

    # Feature 8: count the total number of two/three continuous punctuations
    features.append(len([x for x in bag_of_words if re.search('^([\W]{2,3})$', x) and ' ' not in x]))

    # Feature 9: count the total number of contractions
    features.append(len([x for x in bag_of_words if re.search('([A-Za-z])\'([A-Za-z])+$', x)]))

    # Feature 10: count the total number of parenthesis
    features.append(len([x for x in bag_of_words if x is '(' or x is ')']))

    # Feature 11: count the total number of all capital words
    features.append(len([x for x in bag_of_words if x.isupper()]))

    # Feature 12: count the total number of emoticons
    features.append(emoji.emoji_count(text))

    # Feature 13: count the total number of happy emoticons
    features.append(len([x for x in bag_of_words if ':)' in x or '(:' in x or ':-)' in x or '(-:' in x]))

    # Feature 14: count the total number of sentences without capital letter
    features.append(len([x for x in bag_of_sentences if x[0].islower()]))

    # Feature 15: count the total number of quotation
    features.append(len([x for x in bag_of_words if '\"' in x])//2)

    # Feature 16: count the frequency of small i
    features.append(len([x for x in bag_of_words if re.search('^i$', x)]))

    # Feature 17: count the frequency of full stop without space
    features.append(len(re.findall('\.+[A-Za-z]+', text)))

    # Feature 18: count the frequency of questions
    features.append(len([x for x in bag_of_sentences if '?' in x]))

    # Feature 19: count the total number of sentences with small letter
    features.append(sum([len([x for x in bag_of_sentences if x[0].islower()]), len(re.findall('\.+[a-z]+', text))]))

    # Feature 20: count the frequency of digits
    features.append(len([x for x in bag_of_words if x.isdigit()]))

    # Feature 21: count the frequency of uppercase letters
    features.append(len([x for x in text if x.isupper()]))

    # Feature 22: count the frequency of whitespaces
    features.append(len([x for x in text if x is ' ']))

    # Feature 23: count the frequency of tabs
    features.append(len([x for x in bag_of_sentences if '   ' in x]))

    # Feature 24: count the frequency of error of a
    features.append(len([x for x in re.findall('[A(\sa)]\s+[aeiou][a-z]+', text) if 'a one' not in x.lower() and 'a uni' not in x.lower()]))

    # Feature 25: count the frequency of error of an
    features.append(len(re.findall('[A(\sa)]n\s+[bcdfghjklmnpqrstvwxyz][a-z]+', text)))

    # Features 26-51: count the frequency of all characters [a-z]
    for c in "abcdefghijklmnopqrstuvwxyz":
        features.append(sum([x.lower().count(c) for x in bag_of_words]))

    # Features 52-201: count the frequency of function words
    for w in function_words:
        features.append(len([x for x in bag_of_words if x.lower() == w]))

    # Feature 202-237: count the frequency of parts of speech tags
    for pos in pos_tags:
        features.append(len([x for x, p in nltk.pos_tag(bag_of_words) if p == pos]))

    # Feature 238-247: count the frequency of digits [0-9]
    for d in "0123456789":
        features.append((len([x for x in text if x == d])))

    # Feature 248-279: count the frequency of punctuations
    for p in string.punctuation:
        features.append(len([x for x in text if x == p]))

    # Feature 280: average sentence length (words per sentence)
    features.append(len(bag_of_words) // len(bag_of_sentences))

    # Feature 281: average word length (characters per word)
    features.append(len(text) // len(bag_of_words))

    ''' 
    Feature 282-284: count the frequency of 100 most common uni-, bi- and trigrams of words (excl. stop words) in given book
    Feature 285-286: count the frequency of 100 most common bi- and trigrams of characters in given book
    '''
    # Get all ngrams of current text
    all_ngrams = get_ngrams(text, 3)

    # Create a list of ngrams with their frequency of ngrams that are in top 100 of given book
    freq_ngrams = []
    for i in range(len(all_ngrams)):
        temp = []
        for k, f in common_ngrams[i]:
            temp.append([(key, len(list(freq))) for key, freq in groupby(sorted(all_ngrams[i])) if key == k])
        freq_ngrams.append(chain.from_iterable(temp))

    # Count the total frequency of 100 most common ngrams
    for gram in freq_ngrams:
        features.append(sum([freq for key, freq in gram]))
    ''''''

    return features


# Get a list of ngrams of a book
def get_ngrams(book, number):

    # Get ngrams of words (excl. stop words)
    all_ngrams = []
    for i in range(1, number+1):
        all_ngrams.append([x for x in ngrams([x for x in wordpunct_tokenize(book) if x.lower() not in stop_words], i)])

    # Get ngrams of characters
    for i in range(2, number+1):
        all_ngrams.append([x for x in ngrams(book, i)])

    return all_ngrams


# Get the 100 most common uni-, bi- and trigrams of words and characters
def get_common_ngrams(book):

    # Get all uni-, bi- and trigrams of words and all bi- and trigrams of characters
    all_ngrams = get_ngrams(book, 3)

    # Get list of unique ngrams sorted on descending frequency
    common_ngrams = []
    for gram in all_ngrams:
        common_ngrams.append(sorted([(key, len(list(freq))) for key, freq in groupby(sorted(gram))], key=lambda f: f[1], reverse=True))

    return [x[:100] for x in common_ngrams]


# Get feature lists of the books
def get_featurelists(book):

    # Preparation for topic features: get 100 most common uni-, bi- and trigrams of the given book
    common_ngrams = get_common_ngrams(book)

    # Extract the features of the given book
    features_book = (book_id, extract_features(book, common_ngrams))

    # Create new file and write the features of the given book to it
    path_feat_book = "C:/Users/gebruiker/Documents/Master/Text/Project/output_data/features_book.txt"
    with open(path_feat_book, 'r+', encoding="utf-8") as output_book:
        output_book.write(str(features_book))
        output_book.close()

    # Create new file to write the features of the dataset books to
    path_feat_books = "C:/Users/gebruiker/Documents/Master/Text/Project/output_data/features_dataset.txt"
    output_dataset = open(path_feat_books, 'r+', encoding="utf-8")

    # Extract the features of the dataset books
    features_dataset = []
    for i in IDs:
        features_dataset.append((i, extract_features(strip_headers(load_etext(i)).strip(), common_ngrams)))

        # Write the features to the output file
        output_dataset.write("\n Book " + str(i) + ": ")
        output_dataset.write(str(features_dataset[len(features_dataset)-1]))
    output_dataset.close()

    return features_book, features_dataset


# Compare features and return list of recommended books
def compare_features(book, dataset, number):

    # Calculate the Euclidean distance between the feature lists of the given book and of each dataset book
    distances = []
    for l in dataset:
        distances.append((l[0], scipy.spatial.distance.euclidean(book[1], l[1])))

    # Sort dataset list based on minimal distance
    recommended = sorted(distances, key=lambda dis: dis[1])

    # Create new file and write the list of all recommended books to it
    path_feat_book = "C:/Users/gebruiker/Documents/Master/Text/Project/output_data/recommended_books.txt"
    with open(path_feat_book, 'r+', encoding="utf-8") as output_recommended:
        output_recommended.write(str(recommended))
        output_recommended.close()

    return [recommended[i][0] for i in range(number+1)]


def main():

    # The given book, e.g. A Christmas Carol in Prose; Being a Ghost Story of Christmas - Charles Dickens (ID = 46)
    global book_id
    book_id = 46
    book = strip_headers(load_etext(book_id)).strip()

    # Get features of the given book and the dataset books
    features_book, features_dataset = get_featurelists(book)
    print("All features complete")

    # Get list of [number] recommended books based on the compared features
    recommended = compare_features(features_book, features_dataset, number=5)
    print("Recommended books:")
    print(recommended)


if __name__ == '__main__':
    main()
