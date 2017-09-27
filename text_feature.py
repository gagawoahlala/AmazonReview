import gzip
from collections import defaultdict
import random
import numpy as np
from sklearn.metrics import mean_squared_error
from dask.array.random import beta
import pandas as pd
import operator
import string
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
import nltk
from wordcloud import WordCloud
from nltk.stem.porter import *
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn import linear_model



# nltk.download()
from nltk.corpus import stopwords 
from Crypto.Util.RFC1751 import wordlist



sns.set(style="white", color_codes=True)
    
path = "../amazon-fine-foods/"
food = pd.read_csv(path+"Reviews.csv",header=0)
food.sample(frac=1)
punctuation = set(string.punctuation)

# for word in stopwords.words("english"):
#     punctuation.add(str(word))
#     
# print punctuation

food["length_text"]=food["Text"].str.len()
print food["Score"].value_counts()
print food[:1]
biunigramCount = defaultdict(int)
unigramCount = defaultdict(int)
earliest= food["Time"].min()
food["Time_compress"]=np.around((food["Time"]-earliest)*1.0/31536000)
food=food[:10000]

# helpfulness=food.query("Score == 2 or Score==1")
# helpfulness=food.query("Score == 3 or Score==4 or Score==5")

text=""
punctuation.add("<br />")
# for foodie in food["Summary"]:
#     text += str(foodie).lower()
#     r = ''.join([c for c in foodie.lower() if not c in punctuation ])
#     r = ''.join([c for c in foodie.lower() if c not 'br'])
#     text += r+" "
    
# for foodie in helpfulness["Text"]:
# #     text += foodie.lower()
#     r = ''.join([c for c in foodie.lower() if not c in punctuation ])
# #     r = ''.join([c for c in foodie.lower() if c not 'br'])
#     text += r+" "

for d in food["Text"]:
    r = ''.join([c for c in d.lower() if not c in punctuation])
    bigrams = [b for b in zip(r.split()[:-1], r.split()[1:])]
    r = list([r])
    for w in r[0].split():
        biunigramCount[w] += 1
#     for b in bigrams:
#         biunigramCount[b] += 1
counts = [(biunigramCount[w], w) for w in biunigramCount]
counts.sort()
counts.reverse()
words = [x[1] for x in counts[:1000]]
wordId = dict(zip(words, range(len(words))))
wordSet = set(words)
   
def feature(datum):
    feat = [0]*len(words)
    r = ''.join([c for c in datum.lower() if not c in punctuation])
    bigrams = [b for b in zip(r.split()[:-1], r.split()[1:])]
    r = list([r])
#     for w in bigrams:
#         if w in words:
#             feat[wordId[w]] += 1
    for w in r[0].split():
        if w in words:
            feat[wordId[w]] += 1
    feat.append(1) #offset
    return feat
   
X = [feature(d) for d in food["Text"]]
y = [d for d in food["Score"]]
clf = linear_model.Ridge(1.0, fit_intercept=False)

clf.fit(X, y)
theta = clf.coef_
wordthetalist=[]
for w in wordId:
    wordthetalist.append((w,theta[wordId[w]]))
wordthetalist=sorted(wordthetalist,key=lambda x: x[1], reverse=True)
print wordthetalist[:5]
print wordthetalist[:-6:-1]
#     

magnifiedlist=[]
for word in wordthetalist:
    temp=(word[0],np.around(word[1]*1000))
    magnifiedlist.append(temp)
 

print magnifiedlist[:5]
print magnifiedlist[:-6:-1]   

sentence=[]

for word in magnifiedlist[:1000]:
    tempstring= ""
#     print word[0]
    if type(word[0]) != tuple:
        tempstring=word[0]
        
    else:    
        for smaller in word[0]:
            tempstring+= smaller+' '
    for i in range(int(word[1])):
        sentence.append(tempstring)
    
random.shuffle(sentence)
new_sentence = ' '.join(sentence)
# def scramble(word):
#     foo = list(word)
#     random.shuffle(foo)
#     return ''.join(foo)

# print new_sentence

wordcloud = WordCloud(background_color="white",width=1024,height=768,max_font_size=200).generate(new_sentence)
plt.figure()
plt.imshow(wordcloud)
plt.axis("off")
plt.show()





