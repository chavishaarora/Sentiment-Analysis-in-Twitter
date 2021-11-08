import pandas as pd
import re
from textblob import TextBlob
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords , opinion_lexicon
import numpy as np
from nltk.stem import WordNetLemmatizer
import time
import nltk.sentiment.util
import math
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

data=pd.read_csv("./sentiment_train.csv",header=None)
data=data.drop([1,2,3,4],axis=1)
data.columns=['positivity','sentence']
data=data[1:]
data.reset_index(drop=True, inplace=True)
size=5000
data=data[:size]



lemmatizer=WordNetLemmatizer()
def DataCleaning(data_features):		
	listofsentences=data_features['sentence'].tolist()
	templist=list()
	for i in range(len(listofsentences)):
		tokens=listofsentences[i].lower().split()
		lemmatized_words=[]
		for token in tokens:
			lemma=lemmatizer.lemmatize(token)
			if (lemma!='-PRON-'):
				lemmatized_words.append(lemma)
			else:
				lemmatized_words.append(token)		
		words_1=[]
		for word in lemmatized_words:
			word=str(word)
			if word[:1]=='@' or word[:4]=='http':
				if word[:1]=='@':
					words_1.append('@someuser')
				else:
					words_1.append('http://someurl')
			else:
				words_1.append(word)					
		sentence=' '.join(j for j in words_1)		
		templist.append(sentence)
		# print(i,sentence)	
	data_features['sentence']=templist
	return data_features

def AddPOSTags(clean_data):
	dict_of_tags=dict()
	sentences=clean_data['sentence'].tolist()
	for i in range(len(sentences)):
		words=sentences[i].split()
		pos_tags=nltk.pos_tag(words)
		counts=Counter(tag for word, tag in pos_tags)
		counts=dict(counts)
		for key,value in counts.items():
			if key in dict_of_tags:
				clean_data.set_value(i,key,value)
			else:
				dict_of_tags[key]=int
				clean_data.loc[i,key]=value			
		# print(i, "POS tagging")
	return clean_data.fillna(0),dict_of_tags

def AddNGrams(clean_data):
	sentences=clean_data['sentence'].tolist()
	for i in range(len(sentences)):
		sentences[i]=str(sentences[i])
	vectorizer = CountVectorizer(min_df=15)
	X=vectorizer.fit_transform(sentences)	
	count_vect_df = pd.DataFrame(X.todense(), columns=vectorizer.get_feature_names())	
	# print("unigrams included- ",count_vect_df.shape)
	clean_data=pd.concat([clean_data, count_vect_df], axis=1)
	vectorizer2 = CountVectorizer(analyzer='word', ngram_range=(2, 2),min_df=12)
	X2 = vectorizer2.fit_transform(sentences)	
	count_vect_df_2 = pd.DataFrame(X2.todense(),columns=vectorizer2.get_feature_names())	
	# print("bigrams included- ",count_vect_df_2.shape)
	clean_data=pd.concat([clean_data,count_vect_df_2],axis=1)	
	return clean_data.fillna(0)		

def countAllCaps(sentence):
	tokens=word_tokenize(sentence)
	count=0
	for token in tokens:
		for z in range(len(token)):
			if(len(token)==1):
				break			
			if(token[z].isalpha()):
				if(token[z].islower()):
					break
			if(z==len(token)-1):
				if(token[z].isalpha()):
					if(token[z].isupper()):
						count+=1
	return count

def countemoticons(sentence):
	return len(re.findall(':\)|;\)|:-\)|\(-:|:-D|=D|:P|xD|X-p|\^\^|:-*|\^\.\^|\^\-\^|\^\_\^|\,-\)|\)-:|:\'\(|:\(|:-\(|:\S|T\.T|\.\_\.|:<|:-\S|:-<|\*\-\*|:O|=O|=\-O|O\.o|XO|O\_O|:-\@|=/|:/|X\-\(|>\.<|>=\(|D:', sentence))
def countHashtags(sentence):
	return len(re.findall('#',sentence))
	return "".join(cor_sentence)
def count_elongated(sentence):
	feature_elongatedwords=0
	feature_elongatedwords+=len([word for word in sentence.split() if re.compile(r"(.)\1{2}").search(word)])
	return feature_elongatedwords
def count_negationWords(sentence):
	tokens=word_tokenize(sentence)
	tokens_NEG=nltk.sentiment.util.mark_negation(tokens)	
	count=0
	for token in tokens_NEG:
		if token[-4:]=='_NEG':
			count+=1
	return count
def putPOStags(sentence):
	tokens=word_tokenize(sentence)
	pos_tags=nltk.pos_tag(tokens)	
	return counts

# LLLEEEXXIIICCOONNSSS
from nltk.tokenize import TweetTokenizer
tk = TweetTokenizer() 
fileb=open("./BingLiu.csv").read()
filex=open("./mpqa.txt").read()
filez=open("./unigrams-pmilexicon.txt").read()
fileq=open("./sentiment140_unigram.txt").read()
filey=open('./AFINN-emoticon-8.txt').read()
filew=open('./NRC-word-emotion-lexicon.txt').read()

dict_pos_b=dict()
dict_neg_b=dict()
sentences=fileb.splitlines()
for k,word in enumerate(sentences):
	x=word.split()
	if(x[1]=='positive'):
		dict_pos_b[x[0]]=1
	elif(x[1]=='negative'):
		dict_neg_b[x[0]]=1
def bingliu(sentence):	
	max_score=-2
	last_pos=0
	tokens=tk.tokenize(sentence)
	neg_cnt=0
	pos_cnt=0
	score=0
	for word in tokens:
		if(word in dict_pos_b.keys()):
			pos_cnt+=1
			score=score+dict_pos_b[word]
			if(dict_pos_b[word]>max_score):
				max_score=dict_pos_b[word]
			last_pos=dict_pos_b[word]
		elif(word in dict_neg_b.keys()):
			neg_cnt+=1
			score=score+dict_neg_b[word]
			if(dict_neg_b[word]>max_score):
				max_score=dict_neg_b[word]
	score=pos_cnt-neg_cnt
	return pos_cnt, score, max_score,last_pos

dict_pos_x=dict()
dict_neg_x=dict()
sentences=filex.splitlines()
for k,word in enumerate(sentences):
	x=word.split()
	if(x[1]=='positive'):
		dict_pos_x[x[0]]=1
	elif(x[1]=='negative'):
		dict_neg_x[x[0]]=1
def mpqa(sentence):	
	max_score=-2
	last_pos=0
	tokens=tk.tokenize(sentence)
	neg_cnt=0
	pos_cnt=0
	score=0
	for word in tokens:
		if(word in dict_pos_x.keys()):
			pos_cnt+=1
			score=score+dict_pos_x[word]
			if(dict_pos_x[word]>max_score):
				max_score=dict_pos_x[word]
			last_pos=dict_pos_x[word]
		elif(word in dict_neg_x.keys()):
			neg_cnt+=1
			score=score+dict_neg_x[word]
			if(dict_neg_x[word]>max_score):
				max_score=dict_neg_x[word]
	score=pos_cnt-neg_cnt
	return pos_cnt, score, max_score,last_pos

dict_pos_z=dict()
dict_neg_z=dict()
counter=0
sentences=filez.splitlines()
for k,word in enumerate(sentences):
	x=word.split()			
	if(float(x[1])>=0):		
		dict_pos_z[x[0]]=float(x[1])
	elif(float(x[1])<0):
		dict_neg_z[x[0]]=float(x[1])
def unigrams_pmilexicon(sentence):
	max_score=-2
	last_pos=0
	tokens=tk.tokenize(sentence)
	neg_cnt=0
	pos_cnt=0
	score=0
	for word in tokens:
		if(word in dict_pos_z.keys()):
			pos_cnt+=1
			score=score+dict_pos_z[word]
			if(dict_pos_z[word]>max_score):
				max_score=dict_pos_z[word]
			last_pos=dict_pos_z[word]
		elif(word in dict_neg_z.keys()):
			neg_cnt+=1
			score=score+dict_neg_z[word]
			if(dict_neg_z[word]>max_score):
				max_score=dict_neg_z[word]
	score=pos_cnt-neg_cnt
	return pos_cnt, score, max_score,last_pos

dict_pos_q=dict()
dict_neg_q=dict()
sentences=fileq.splitlines()
for k,word in enumerate(sentences):
	x=word.split()			
	if(float(x[1])>=0):		
		dict_pos_q[x[0]]=float(x[1])
	elif(float(x[1])<0):		
		dict_neg_q[x[0]]=float(x[1])
def sentiment140_unigram(sentence):
	max_score=-2
	last_pos=0
	tokens=tk.tokenize(sentence)
	neg_cnt=0
	pos_cnt=0
	score=0
	for word in tokens:
		if(word in dict_pos_q.keys()):
			pos_cnt+=1
			score=score+dict_pos_q[word]
			if(dict_pos_q[word]>max_score):
				max_score=dict_pos_q[word]
			last_pos=dict_pos_q[word]
		elif(word in dict_neg_q.keys()):
			neg_cnt+=1
			score=score+dict_neg_q[word]
			if(dict_neg_q[word]>max_score):
				max_score=dict_neg_q[word]
	return pos_cnt, score, max_score,last_pos

dict_pos_y=dict()
dict_neg_y=dict()
sentences=filey.splitlines()
for k,word in enumerate(sentences):
	x=word.split()
	if(int(x[1])>=0):
		dict_pos_y[x[0]]=int(x[1])
	elif(int(x[1])<0):
		dict_neg_y[x[0]]=int(x[1])
def AFINN_emoticon(sentence):
	max_score=-2
	last_pos=0
	tokens=tk.tokenize(sentence)
	neg_cnt=0
	pos_cnt=0
	score=0
	for word in tokens:
		if(word in dict_pos_y.keys()):
			pos_cnt+=1
			score=score+dict_pos_y[word]
			if(dict_pos_y[word]>max_score):
				max_score=dict_pos_y[word]
			last_pos=dict_pos_y[word]
		elif(word in dict_neg_y.keys()):
			neg_cnt+=1
			score=score+dict_neg_y[word]
			if(dict_neg_y[word]>max_score):
				max_score=dict_neg_y[word]
	score=pos_cnt-neg_cnt
	return pos_cnt, score, max_score,last_pos

dict_pos_w=dict()
dict_neg_w=dict()
sentences=filew.splitlines()
for k,word in enumerate(sentences):
	x=word.split()
	if(x[2]=='1'):
		if(x[1]=='anticipation' or x[1]=='joy' or x[1]=='positive' or x[1]=='surprise' or x[1]=='trust'):
			dict_pos_w[x[0]]=1
		else:
			dict_pos_w[x[0]]=1
def NRC_emotion(sentence):
	max_score=-2
	last_pos=0
	tokens=tk.tokenize(sentence)
	neg_cnt=0
	pos_cnt=0
	score=0
	for word in tokens:
		if(word in dict_pos_w.keys()):
			pos_cnt+=1
			score=score+dict_pos_w[word]
			if(dict_pos_w[word]>max_score):
				max_score=dict_pos_w[word]
			last_pos=dict_pos_w[word]
		elif(word in dict_neg_w.keys()):
			neg_cnt+=1
			score=score+dict_neg_w[word]
			if(dict_neg_w[word]>max_score):
				max_score=dict_neg_w[word]
	score=pos_cnt-neg_cnt
	return pos_cnt, score, max_score,last_pos

def checklasttokenpunctuation(sentence):
	tokens=tk.tokenize(sentence)
	if (tokens[-1]=='!' or tokens[-1]=='?'):
		return 1
	else:
		return 0

def contiguouspunctuation(sentence):	
	count = len(re.findall('[!?]+', sentence))
	return count

dict_of_tags=dict()
columns_list=list()
def addfeatures(data_features,flag,dict_of_tags,columns_list): 
	# 2) Character n-grams (No need to consider)
	data_features['Num_AllCaps']=data_features.apply(lambda row:countAllCaps(row['sentence']),axis=1)	# 3) count number of words with all caps
	data_features['Num_Hashtags']=data_features.apply(lambda row:countHashtags(row['sentence']),axis=1)		# 5) count number of hashtags	
	data_features['contiguouspunctuation']=data_features.apply(lambda row:contiguouspunctuation(row['sentence']),axis=1)	# 7) Punctuation (a)
	data_features['Last_token_punctuation']=data_features.apply(lambda row:checklasttokenpunctuation(row['sentence']),axis=1) # 7) Punctuation (b)
	data_features['Num_Emoticons']=data_features.apply(lambda row:countemoticons(row['sentence']),axis=1)	# 8) Number of Emoticons
	# 8) Need to do more on Emoticons 
	data_features['Num_ElongatedWords']=data_features.apply(lambda row:count_elongated(row['sentence']),axis=1)		# 9) Count number of elongated words
	# 10) Clusters have to be skipped	
	data_features['Num_NegationWords']=data_features.apply(lambda row:count_negationWords(row['sentence']),axis=1)	# 11) count number of negative words _NEG	
	# 6) Lexicons
	x=data_features.apply(lambda row:bingliu(row['sentence']),axis=1)
	x=list(x)
	data_features=data_features.join(pd.DataFrame(x,columns=['pos_cnt_b','score_b','max_score_b','last_pos_b']))

	x=data_features.apply(lambda row:sentiment140_unigram(row['sentence']),axis=1)
	x=list(x)
	data_features=data_features.join(pd.DataFrame(x,columns=['pos_cnt_q','score_q','max_score_q','last_pos_q']))

	x=data_features.apply(lambda row:mpqa(row['sentence']),axis=1)
	x=list(x)
	data_features=data_features.join(pd.DataFrame(x,columns=['pos_cnt_x','score_x','max_score_x','last_pos_x']))

	x=data_features.apply(lambda row:unigrams_pmilexicon(row['sentence']),axis=1)
	x=list(x)
	data_features=data_features.join(pd.DataFrame(x,columns=['pos_cnt_z','score_z','max_score_z','last_pos_z']))

	x=data_features.apply(lambda row:AFINN_emoticon(row['sentence']),axis=1)
	x=list(x)
	data_features=data_features.join(pd.DataFrame(x,columns=['pos_cnt_y','score_y','max_score_y','last_pos_y']))

	x=data_features.apply(lambda row:NRC_emotion(row['sentence']),axis=1)
	x=list(x)
	data_features=data_features.join(pd.DataFrame(x,columns=['pos_cnt_w','score_w','max_score_w','last_pos_w']))	

	if flag==0:
		clean_data=DataCleaning(data_features)
		clean_data,dict_of_tags=AddPOSTags(data_features)		# 4) Number of occurance of POS tag		
		data_features=AddNGrams(clean_data)		# 1) Word n-grams		
		columns_list=list(data_features.columns)
		columns_list=columns_list[33:]
	else:				
		clean_data_test=DataCleaning(data_features)		
		data_features=pd.concat([clean_data_test,pd.DataFrame(columns=columns_list)],axis=1)
		sentences=data_features['sentence']
		columns_set=set(columns_list)
		for i in range(len(sentences)):
			words=sentences[i].split()
			pos_tags=nltk.pos_tag(words)			
			counts=Counter(tag for word, tag in pos_tags)
			counts=dict(counts)
			for key,value in counts.items():
				if key in dict_of_tags:
					data_features.set_value(i,key,value)
			for j in range(len(words)):
				if words[j] in columns_set:
					data_features.set_value(i,words[j],1)
			for j in range(len(words)-1):
				bigram_word=words[j]+' '+words[j+1]
				if bigram_word in columns_set:
					data_features.set_value(i,bigram_word,1)
			# print(i)
		data_features.fillna(0)
	
	return data_features.drop(['sentence'],axis=1), dict_of_tags,columns_list

data_features,dict_of_tags,columns_list=addfeatures(data,0,dict_of_tags,columns_list)

data_test=pd.read_csv("./sentiment_test.csv",header=None)
data_test=data_test.drop([1,2,3,4],axis=1)
data_test.columns=['positivity','sentence']
data_test=data_test[1:]
data_test.reset_index(drop=True, inplace=True)
data_test=data_test[:size]
data_features_test,dict_of_tags,columns_list=addfeatures(data_test,1,dict_of_tags,columns_list)
data_features_test=data_features_test.fillna(0)
# print("data_features_test matrix-\n",data_features_test)

#use pca from sklearn to reduce dimensionality, for svm
X_train=data_features[data_features.columns[1:]]
# print("X_train matrix\n",X_train)
y_train=data_features['positivity']
X_test=data_features_test[data_features_test.columns[1:]]
y_test=data_features_test['positivity']

#-------------------------------------------- Machine Learning Algorithms --------------------------------------------
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import seaborn as sns
def evaluate(y_true, y_pred, mode='macro'):
  print('Accuracy:',accuracy_score(y_true, y_pred))
  print("Precision-Recall-F1 Score", precision_recall_fscore_support(y_true, y_pred, average=mode)[:3])
  #confusion matrix
  cm = metrics.confusion_matrix(y_true, y_pred)
  print('----'*10)
  print(cm)
  print('----'*10)
  print('\n')
  ##Annotation
  sns.heatmap(cm, annot=True)

#---------------------- Naive Bayes ------------------- 
vocabulary_set=set()
listofsentences=data['sentence'].tolist()
for i in listofsentences:
	tokens=tk.tokenize(i)
	for token in tokens:
		vocabulary_set.add(token)
listofsentences_test=data_test['sentence'].tolist()
for i in listofsentences_test:
	tokens=tk.tokenize(i)
	for token in tokens:
		vocabulary_set.add(token)
vocabulary_list=list(vocabulary_set)
nb_matrix=pd.DataFrame(columns=vocabulary_list,index=[4,0])
nb_matrix=nb_matrix.fillna(0)
v=len(vocabulary_list)
index=0
num_of_4=0
num_of_0=0
for i in listofsentences:
	tokens=tk.tokenize(i)
	positivity=data.get_value(index,'positivity')
	for token in tokens:		
		nb_matrix.set_value(positivity,token, nb_matrix.get_value(positivity,token) + 1)
	index+=1
	if positivity==4:
		num_of_4+=1
	else:
		num_of_0+=1

nb_matrix=nb_matrix.astype('float64')
for token in vocabulary_list:
	count_4=nb_matrix.get_value(4,token)
	count_0=nb_matrix.get_value(0,token)
	prob_4=(count_4+1)/float(num_of_4+v)
	prob_0=(count_0+1)/float(num_of_0+v)
	nb_matrix.set_value(4,token,prob_4)
	nb_matrix.set_value(0,token,prob_0)

y_pred=[]
for i in listofsentences_test:
	tokens=tk.tokenize(i)
	probability_4=float(0)
	probability_0=float(0)
	for token in tokens:
		probability_4+=math.log(nb_matrix.get_value(4,token))
		probability_0+=math.log(nb_matrix.get_value(0,token))
	probability_4+=math.log(num_of_4/(num_of_4+num_of_0))
	probability_0+=math.log(num_of_0/(num_of_4+num_of_0))	
	if probability_4>probability_0:
		y_pred.append(4)
	else:
		y_pred.append(0)
print('-'*20,'Results of Naive Bayes- ')
evaluate(y_test,y_pred)
#---------------------- SVM ------------------- 
from sklearn.svm import LinearSVC
#Linear SVC Classifier
clf = LinearSVC(penalty='l2', loss='squared_hinge', C=100, dual=False)
# Fitting training data into classifier
clf.fit(X_train,y_train)
# print('Training accuracy: {:.2f}'.format(clf.score(X_train, y_train) * 100))

y_pred = clf.predict(X_test)
print('-'*20,'Results of SVM- ')
evaluate(y_test, y_pred)
#---------------------- Decision Tree ------------------- 
from sklearn.tree import DecisionTreeClassifier
# DT Classifier
dt = DecisionTreeClassifier()

# Lets fit the data into classifier 
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
print('-'*20,'Results of DecisionTreeClassifiern- ')
evaluate(y_test, y_pred)
#---------------------- MLP ------------------- 
from sklearn.neural_network import MLPClassifier # neural network
# Classifier
clf = MLPClassifier(alpha=1e-5, hidden_layer_sizes=(3, 3), random_state=1, verbose=True, max_iter=50)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print('-'*20,'Results of MLPClassifier- ')
evaluate(y_test, y_pred)


