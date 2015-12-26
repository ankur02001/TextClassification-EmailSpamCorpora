# program to get started creating a spam detection classifier
# open python and nltk packages needed for processing
import os
import sys
import random
import nltk
from nltk.corpus import stopwords
import csv
import re
import math
import string
from nltk.tokenize import wordpunct_tokenize as tokenize
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from itertools import islice, izip
from collections import Counter

# read stop words from file if used
stopwords_email = [line.strip() for line in open('stopwords_emailSpam.txt')]

# define a feature definition function here
#########################################################################################
####  Pre-processing the documents  ####
#########################################################################################
def Pre_processing_documents(document):
	# "Pre_processing_documents"  
	# "create list of lower case words"
	word_list = re.split('\s+', document.lower())
	# punctuation and numbers to be removed
	punctuation = re.compile(r'[-.?!/\%@,":;()|0-9]')
	word_list = [punctuation.sub("", word) for word in word_list] 
	final_word_list = []
	for word in word_list:
		if word not in stopwords_email:
			final_word_list.append(word)
	stringword = " ".join(final_word_list)
	return stringword 




#########################################################################################
####  Features   accuracy calculation####
#########################################################################################
def Features_accuracy_calculation(featuresets):
	print "---------------------------------------------------"
	print "Training and testing a classifier "  
	training_size = int(0.1*len(featuresets))
	test_set = featuresets[:training_size]
	training_set = featuresets[training_size:]
	classifier = nltk.NaiveBayesClassifier.train(training_set)
	print "Accuracy of classifier :"
	print nltk.classify.accuracy(classifier, test_set)
	print "---------------------------------------------------"
	print "Showing most informative features"
	print classifier.show_most_informative_features(50)
	print "---------------------------------------------------"
	print "Obtaining precision, recall and F-measure scores"
	Obtain_precision_recall_and_Fmeasure_scores(classifier,test_set)
	print ""

	
	
	
	
#########################################################################################
## Obtain precision, recall and F-measure scores. ##
#########################################################################################
def Obtain_precision_recall_and_Fmeasure_scores(classifier_type, test_set):
  reflist = []
  testlist = []
  for (features, label) in test_set:
	reflist.append(label)
	testlist.append(classifier_type.classify(features))
  print " "
  print "The confusion matrix"
  cm = nltk.metrics.ConfusionMatrix(reflist, testlist)
  print cm

#  precision and recall
# start with empty sets for true positive, true negative, false positive, false negative,

  (refpos, refneg, testpos, testneg) = (set(), set(), set(), set())
  
  for i, label in enumerate(reflist):
	if label == 'spam': refneg.add(i)
	if label == 'ham': refpos.add(i)
  for i, label in enumerate(testlist):
	if label == 'spam': testneg.add(i)
	if label == 'ham': testpos.add(i)
  
  def printmeasures(label, refset, testset):
	print label, 'precision:', nltk.metrics.precision(refset, testset)
	print label, 'recall:', nltk.metrics.recall(refset, testset)
	print label, 'F-measure:', nltk.metrics.f_measure(refset, testset)
  
  printmeasures('Positive_HAM ', refpos, testpos)
  print ""
  printmeasures('Negative_SPAM ', refneg, testneg)
  print ""

  
  
  
  
  
  
#########################################################################################
## cross-validation ##
#########################################################################################
# this function takes the number of folds, the feature sets
# it iterates over the folds, using different sections for training and testing in turn
#   it prints the accuracy for each fold and the average accuracy at the end
def cross_validation(num_folds, featuresets):
	subset_size = len(featuresets)/num_folds
	accuracy_list = []
	print "Running cross_validation for classifier :"
	# iterate over the folds
	for i in range(num_folds):
		print "---------------------------------------------------"
		test_this_round = featuresets[i*subset_size:][:subset_size]
		train_this_round = featuresets[:i*subset_size] + featuresets[(i+1)*subset_size:]
        # train using train_this_round
		classifier = nltk.NaiveBayesClassifier.train(train_this_round)
        # evaluate against test_this_round and save accuracy
		accuracy_this_round = nltk.classify.accuracy(classifier, test_this_round)
		print 'Accuracy Round ', i, ' = ' ,  accuracy_this_round
		Obtain_precision_recall_and_Fmeasure_scores(classifier,test_this_round)
		accuracy_list.append(accuracy_this_round)
	# find mean accuracy over all rounds
	print 'Mean accuracy over all rounds = ', sum(accuracy_list) / num_folds
 

  
#########################################################################################
## unigram word features as a baseline ##
#########################################################################################

csv_file = open('unigramWordFeatures.csv', 'wb')
unigram_writer = csv.writer(csv_file, quoting = csv.QUOTE_ALL)

def get_unigram_word_features(all_words):  
  print "---------------------------------------------------"
  print "Getting all words and create word features"
  words =  nltk.FreqDist(w.lower() for w in all_words)
  word_features = words.keys()[:100]
  unigram_csv_features = word_features
  unigram_csv_features.append("Category")
  unigram_writer.writerow(unigram_csv_features)
  return word_features
  
  
  
def get_unigram_word_features_sets(email, category,word_features):
	email_words = set(email);
	features = {}
	email_csv = []
	for word in word_features:
		features['contains(%s)'% word] = word in email_words
		if word == 'Category':
			email_csv.append(category)
		elif word in email_words:
			email_csv.append("true")
		else:
			email_csv.append("false")
	unigram_writer.writerow(email_csv)
	return features
	
	
#########################################################################################

## adding Bigram features ##
#########################################################################################	

# set up for using bigrams
from nltk.collocations import *
bigram_measures = nltk.collocations.BigramAssocMeasures()


from itertools import tee, islice

def ngrams(lst, n):
  tlst = lst
  while True:
    a, b = tee(tlst)
    l = tuple(islice(a, n))
    if len(l) == n:
      yield l
      next(b)
      tlst = b
    else:
      break
	  
def get_bigram_word_features(hamtexts , spamtexts):  
  print "---------------------------------------------------"
  print "Getting all words and create word features"
  # create the bigram finder on the movie review words in sequence
  words = ""
  for spam in hamtexts:
	spam1 = Pre_processing_documents(spam)
	words += spam1
  
  for spam in spamtexts:
	spam1 = Pre_processing_documents(spam)
	words += spam1  
	
  bigram_measures = nltk.collocations.BigramAssocMeasures()
  finder = BigramCollocationFinder.from_words(words.split(),window_size = 4)
  print finder
  finder.apply_freq_filter(6)
  top20 = finder.nbest(bigram_measures.pmi,3000)  
  bigram_features = finder.nbest(bigram_measures.chi_sq, 3000)
  print " Applying bigram_measures.pmi"
  #print top20[:20]
  print " Applying bigram_measures.chi_sq"
  #print bigram_features[:20]
  return bigram_features[:5000]




# define features that include words as before 
# add the most frequent significant bigrams
# this function takes the list of words in a document as an argument and returns a feature dictionary
# it depends on the variables word_features and bigram_features
def bigram_document_features(document_b,c,unigram_word_feature_b,bigram_word_feature_b):
  document_words = set(document_b)
  document_bigrams = nltk.bigrams(document_b)
  features = {}
  #print "hello1"
  for word_itr in unigram_word_feature_b:
	#print word_itr
	features['contains(%s)' % word_itr] = word_itr in document_words
	
  for bigram_itr in bigram_word_feature_b:
	#print bigram_itr
	features['bigram(%s %s)' % bigram_itr] = (bigram_itr in document_bigrams)    
  return features
	
	
#########################################################################################
# use word frequency or tfidf scores as the values of the word features, instead of Boolean values
#########################################################################################

tfidf_csv_file = open('tfidfWordFeatures.csv', 'wb')
tfidf_writer = csv.writer(tfidf_csv_file, quoting = csv.QUOTE_ALL)
 
def get_tfidf_word_features_sets(email, category,bigram_word_feature,unigram_word_feature,emaildocs):
	email_words = set(email);
	document_bigrams = nltk.bigrams(email)
	features = {}
	email_csv = []
	weakPos = 0
	for word in bigram_word_feature:
		score = tf_idf(word, email, emaildocs)
		features['bigram(%s %s)' % word] = score
		if word == 'Category':
			email_csv.append(category)
		elif word in email_words:
			email_csv.append("true")
		else:
			email_csv.append("false")
			
	for word in unigram_word_feature:
		score = tf_idf(word, email, emaildocs)
		features['unigram(%s)'% word] = score
		if word == 'Category':
			email_csv.append(category)
		elif word in email_words:
			email_csv.append("true")
		else:
			email_csv.append("false")
	tfidf_writer.writerow(email_csv)
	return features

#########################################################################################
# Calculating  tfidf scores
#########################################################################################
def freq(word, doc):
    return doc.count(word)


def word_count(doc):
    return len(doc)


def tf(word, doc):
    return (freq(word, doc) / float(word_count(doc)))


def num_docs_containing(word, list_of_docs):
    count = 0
    for document in list_of_docs:
        if freq(word, document) > 0:
            count += 1
    return 1 + count


def idf(word, list_of_docs):
    return math.log(len(list_of_docs) /
            float(num_docs_containing(word, list_of_docs)))

def tf_idf(word, doc, list_of_docs):
    return (tf(word, doc) * idf(word, list_of_docs))

	





# function to read spam and ham files, train and test a classifier 
def processspamham(dirPath,limitStr):
  # convert the limit argument from a string to an int
  limit = int(limitStr)
  print ""
  print ""
  # start lists for spam and ham email texts
  hamtexts = []
  spamtexts = []
  os.chdir(dirPath)
  # process all files in directory that end in .txt up to the limit
  for file in os.listdir("./spam"):
    if (file.endswith(".txt")) and (len(spamtexts) < limit):
      # open file for reading and read entire file into a string
      f = open("./spam/"+file, 'r')
      spamtexts.append (f.read())
      f.close()
  for file in os.listdir("./ham"):
    if (file.endswith(".txt")) and (len(hamtexts) < limit):
      # open file for reading and read entire file into a string
      f = open("./ham/"+file, 'r')
      hamtexts.append (f.read())
      f.close()
	  
  print "##############################################"
  print "number emails read"
  print "Number of spam files:",len(spamtexts)
  print "Number of ham files:",len(hamtexts)
  print
  
  print ""
  print ""
  print ""
  print ""
  print "##############################################"
  print "unigram word features as a baseline"
  print "##############################################"

  print "Pre-processing or filtering"
  punctuation_tokenizer = RegexpTokenizer(r'\w+')
  
  # create list of mixed spam and ham email documents as (list of words, label)
  emaildocs = []
  all_words = []
  
  print "---------------------------------------------------"
  print "Add all the spam"
  for spam in spamtexts:
	spam1 = Pre_processing_documents(spam)
	tokens = nltk.word_tokenize(spam1)
	emaildocs.append((tokens, 'spam'))
	for w in tokens:
		all_words.append(w)
  
  print "---------------------------------------------------"
  print "Add all the regular emails"
  for ham in hamtexts:
	ham1 = Pre_processing_documents(ham)
	tokens = nltk.word_tokenize(ham1)
	emaildocs.append((tokens, 'ham'))
	for w in tokens:
		all_words.append(w)
  
  print "---------------------------------------------------"
  print "randomize the list"
  random.shuffle(emaildocs)
  
  print "---------------------------------------------------"
  print "Possibly filter tokens"
  
  
  print "---------------------------------------------------"
  print "continue as usual to get all words and create word features"
  
  unigram_word_feature = get_unigram_word_features(all_words)

  
  print "---------------------------------------------------"
  print "Feature sets from a feature definition function"
  
  unigram_word_featuresets = [(get_unigram_word_features_sets(e,c,unigram_word_feature), c) for (e,c) in emaildocs]

  
  
  print "---------------------------------------------------"
  Features_accuracy_calculation(unigram_word_featuresets)
 
  print ""
  print ""
  print ""
  print ""
  print "##############################################"
  print "Perform the cross-validation on the feature sets with word features"
  print "##############################################"

  num_folds = 5
  #cross_validation(num_folds, unigram_word_featuresets)

  print ""
  print ""
  print ""
  print ""
  print "##############################################"
  print "Adding bigram feature to unigram "
  print "##############################################"
  
  bigram_word_feature = get_bigram_word_features(hamtexts , spamtexts)
  #bigram_featuresets = [(bigram_document_features(e,c,unigram_word_feature,bigram_word_feature), c) for (e,c) in emaildocs]
  #Features_accuracy_calculation(bigram_featuresets)
  #cross_validation(num_folds, bigram_featuresets)

  
  print ""
  print ""
  print ""
  print ""
  print "##############################################"
  print "tfidf scores as the values of the word features, instead of Boolean values"
  print "##############################################"
  
  tfidf_word_featuresets = [(get_tfidf_word_features_sets(e,c,bigram_word_feature,unigram_word_feature,emaildocs), c) for (e,c) in emaildocs]
  Features_accuracy_calculation(tfidf_word_featuresets)
  cross_validation(num_folds, tfidf_word_featuresets)

"""
commandline interface takes a directory name with ham and spam subdirectories
   and a limit to the number of emails read each of ham and spam
It then processes the files and trains a spam detection classifier.

"""
if __name__ == '__main__':
    if (len(sys.argv) != 3):
        print 'usage: classifySPAM.py <corpus-dir> <limit>'
        sys.exit(0)
    processspamham(sys.argv[1], sys.argv[2])
        
