import csv
import nltk
import re
import operator
from copy import deepcopy
import sys
import numpy as np

num_of_topics = 5

#getting top 10 topics from each of the columns
for column_num in range(1, 9):

	print "Top topics for column number:", column_num

	def read_from_CSV(fileName):

		allData = []   #list to hold all the data
		
		with open(fileName, 'rb') as f:

			reader = csv.reader(f)
			count = 0
			
			for row in reader:
				count = count + 1

				#skipping the header row
				if count == 1:
					continue

				#saving only the feedback column
				#not taking in the "NA" columns

				if (row[column_num].strip().lower() != "na" and row[column_num].strip().lower() != "n/a" and row[column_num].strip().lower() != "none") and (row[9] == "Apex+"):
					
					allData.append(unicode(row[column_num].strip(), encoding='utf8', errors='replace'))
		
		#remove repetitions from the user reviews
		allData = list(set(allData))

		return allData

	def clean_data(sentences):

		#remove stopwords
		STOPWORDS = list(set([unicode(line.strip(), encoding='utf8', errors='replace')
		            for line in open("input.txt")]))
		#word tokenize each of the sentences
		sentences = [nltk.word_tokenize(sent) for sent in sentences]
		# print sentences

		clean_sentences = []  #list to hold all sentences

		for sent in sentences:

			no_stopwords = [word.lower() for word in sent if word.lower() not in STOPWORDS]

			clean_sentences.append(no_stopwords)

		return clean_sentences

	#regular expression for extracting nouns from text
	NOUNS = r"N.*"
	CONJUNCTION = r'CC'
	PREPOSITION = r'IN'
	ADJECTIVES = r"JJ.*"
	VERBS = r"VB.*"
	PUNCTUATION = r'PN'
	DETERMINER = r'DT'
	ADVERB = r'RB.*'


	def leaves(tree):
		
		# Finds NP leaf nodes of a chunk tree
		for subtree in tree.subtrees(filter = lambda t : t.label()=='NP'):
			yield subtree.leaves()


	def getNoun(sentences):

		# list of keywords for each comment
		keywords = list()

		#keep a count of the sentence
		count = 0

		#for each sentence in the comment
		for sent in sentences:

			#for each word in the sentence
			for word in sent:

				#if the word is a noun 
				if re.match(NOUNS, word[1]):
					try:
						#for each sentence saving a sentence of keywords
						keywords[count] = keywords[count] + ' ' + word[0]
					
					except Exception, e:
						keywords.append(word[0])

				# if the word is a conjuction, split the sentence on that
				if re.match(CONJUNCTION, word[1]):
					#start a new sentence
					count += 1     

				# if the word is a prepostion, split the sentence on that
				if re.match(PREPOSITION, word[1]):
					# start a new sentence
					count += 1

				# if the word is a conjuction, split the sentence on that
				if re.match(ADJECTIVES, word[1]):
					#start a new sentence
					count += 1     

				# if the word is a prepostion, split the sentence on that
				if re.match(PUNCTUATION, word[1]):
					# start a new sentence
					count += 1

				# if the word is a prepostion, split the sentence on that
				if re.match(VERBS, word[1]):
					# start a new sentence
					count += 1

				# if the word is a prepostion, split the sentence on that
				if re.match(DETERMINER, word[1]):
					# start a new sentence
					count += 1

				# if the word is a prepostion, split the sentence on that
				if re.match(ADVERB, word[1]):
					# start a new sentence
					count += 1


			count += 1

		#returning a string of keywords for each sentence of each of the comments
		return keywords


	def getTermFrequency(corpus):
		
		from sklearn.feature_extraction.text import CountVectorizer 

		# get the entire corpus as a list of list of list
		corpus_list = corpus.values()
		# corpus_list_new = []
		
		#dictionary to save term frequency for each of the words
		termFreqScore = dict()

		# from nested lists, convert to only a list of comment strings
		for i, comment in enumerate(corpus_list):

			for j, sent in enumerate(comment):

				sentence = ' '.join(sent)

				corpus_list[i][j] = sentence

			corpus_list[i] = ' '.join(corpus_list[i])

		# remove empty elements
		vectorizer = CountVectorizer(min_df=1)

		X = vectorizer.fit_transform(corpus_list)
		#mapping of terms to their respective counts

		# print np.asarray(X.sum(axis=0))
		termFreqScore = dict(zip(vectorizer.get_feature_names(), np.asarray(X.sum(axis=0)).ravel()))	
		
		# print termFreqScore

		# print termFreqScore
		# convert matrix to python array
		termCounts = X.toarray()

		# get vocabulary of the enrire corpus
		vocabulary = vectorizer.get_feature_names()

		return corpus_list, termCounts, vocabulary, termFreqScore

	def tokenize(text):

		tokens = nltk.word_tokenize(text)
		return tokens

	def getTfIdfScore(corpus):

		from sklearn.feature_extraction.text import TfidfVectorizer
		tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
		tfs = tfidf.fit_transform(corpus)

		feature_names = tfidf.get_feature_names()

		tfidfScore = {}

		for doc in corpus:
			
			response = tfidf.transform([doc])

			for col in response.nonzero()[1]:

				tfidfScore[feature_names[col]] = response[0, col]

		return tfidfScore

	fileName = sys.argv[1:][0]   #get the filename from the commandline
	documents = read_from_CSV(fileName)  #read the csv file containing the product reviews

	# dictionary to hold the comments after extraction of POS tags and removal of stopwords
	cleanedCorpus = dict()

	for i, comment in enumerate(documents):

		#tokenizing text till word tokenize
		sentences = nltk.sent_tokenize(comment)
		sentences = [nltk.word_tokenize(sent) for sent in sentences]
		sentences = [nltk.pos_tag(sent) for sent in sentences]

		#get nouns from text
		extractedKeys = getNoun(sentences)

		# remove stopwords from extracted nouns
		cleanedStopwords = clean_data(extractedKeys)

		#save topics for each comment
		cleanedCorpus["Comment " + str(i)] = cleanedStopwords
		
	# print cleanedCorpus
	finalCorpus = deepcopy(cleanedCorpus)

	#get word frequency from the entire corpus
	corpus, termCounts, terms, termfreq_score = getTermFrequency(cleanedCorpus)

	#tfidf score for all the words in te corpus
	tfidf_score = getTfIdfScore(corpus)

	#overall topics with highest scores
	sentence_sum_score = {}

	for key in finalCorpus:
		#iterating through each sentence in the comment
		for sentence in cleanedCorpus[key]:

			sentence_list = list(set(sentence.split()))

			sentence_score = []    #maintain sentence score on each sentence

			# if len(sentence_list) < 5:
			for word in sentence_list:

				try:#get a list of term frequency score for each word in the sentence
					sentence_score.append(termfreq_score[word])

				except Exception, e:
					# if word does not have a term frequency score assigned to it
					# print e
					sentence_score.append(0)

			#get score for the entire sentence
			sum_score = sum(sentence_score)
			
			sentence_sum_score[sentence] = sum_score


	#sort score in decreasing order
	sorted_sentence_sum_score = sorted(sentence_sum_score.items(), key=operator.itemgetter(1), reverse=True)

	count = 0

	#print top 20 topics for the NPS dataset
	for key in sorted_sentence_sum_score:
		
		if count > (num_of_topics - 1):
			break
		
		print key[0]

		count = count + 1

	print

	