import re
from bs4 import BeautifulSoup
import distance
from fuzzywuzzy import fuzz
import pickle
import numpy as np

pickle_off = open("word2tfidf.pkl","rb")
word2tfidf = pickle.load(pickle_off)

def preprocess1(q):
    
    q = str(q).lower().strip()
    
    # replacing certain special characters with their string equivalents
    q = q.replace('%', ' percent')
    q = q.replace('$', ' dollar ')
    q = q.replace('₹', ' rupee ')
    q = q.replace('€', ' euro ')
    q = q.replace('@', ' at ')
    
    #expanding contractipns
    # https://stackoverflow.com/a/19794953
    contractions = { 
    "ain't": "am not",
    "aren't": "are not",
    "can't": "can not",
    "can't've": "can not have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have"
    }

    q_decontracted = []

    for word in q.split():
        if word in contractions:
            word = contractions[word]

        q_decontracted.append(word)

    q = ' '.join(q_decontracted)
    q = q.replace("'ve", " have")
    q = q.replace("n't", " not")
    q = q.replace("'re", " are")
    q = q.replace("'ll", " will")
    
    # Removing HTML tags
    q = BeautifulSoup(q)
    q = q.get_text()
    
    # Remove punctuations
    pattern = re.compile('\W')
    q = re.sub(pattern, ' ', q).strip()

    
    return q
  
  def countCommonWords(q1,q2):
    w1 = set(map(lambda word: word.lower().strip(), q1.split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), q2.split(" ")))    
    return len(w1 & w2)
  
  def totalNumberOfWords(q1,q2):
    w1 = set(map(lambda word: word.lower().strip(), q1.split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), q2.split(" ")))    
    return (len(w1) + len(w2))
  
  def findFuzzyFeatures(q1,q2):
    fuzzyFeatures = [0.0]*4
    fuzzyFeatures[0] = fuzz.QRatio(q1, q2)
    fuzzyFeatures[1] = fuzz.partial_ratio(q1, q2)
    fuzzyFeatures[2] = fuzz.token_sort_ratio(q1, q2)
    fuzzyFeatures[3] = fuzz.token_set_ratio(q1, q2)
    return fuzzyFeatures
  
  def findW2vector(q):
  nlp = spacy.load('en_core_web_lg')
  pickle_off = open("word2tfidf.pkl","rb")
  word2tfidf = pickle.load(pickle_off)
  #print(emp)
  vecs1 = []
  #for qu1 in tqdm(list(df['question1'])):
  doc1 = nlp(q) 
      # 384 is the number of dimensions of vectors 
  mean_vec1 = np.zeros([len(doc1), 300])
  for word1 in doc1:
          # word2vec
      vec1 = word1.vector
          # fetch df score
      try:
          idf = word2tfidf[str(word1)]
      except:
          idf = 0
          # compute final vec
      #print(idf)    
      mean_vec1 += vec1 * idf
  mean_vec1 = mean_vec1.mean(axis=0)
  vecs1.append(mean_vec1)
  #print(mean_vec1)
  #return list(vecs1)
  return list(mean_vec1)


def computeQueryPoint(q1,q2):
    query = []

    q1 = preprocess1(q1)
    q2 = preprocess1(q2)
    
    #basic features of the query - 7
    query.append(len(q1))
    query.append(len(q2))
    query.append(len(q1.split(" ")))
    query.append(len(q2.split(" ")))
    query.append(countCommonWords(q1,q2))
    # query.append(totalNumberOfWords(q1,q2))
    query.append(round(countCommonWords(q1,q2)/totalNumberOfWords(q1,q2),2))
    
    # fetch fuzzy features - 4
    fuzzyFeatures = findFuzzyFeatures(q1,q2)
    query.extend(fuzzyFeatures)
    
    # nlp features
    #q1w2v = np.array(findW2vector(q1))
    #q2w2v = np.array(findW2vector(q2))
    q1w2v = findW2vector(q1)
    q2w2v = findW2vector(q2)
    
    #return np.hstack((np.array(query).reshape(1,11),q1w2v,q2w2v))
    query.extend(q1w2v)
    query.extend(q2w2v)
    return query
