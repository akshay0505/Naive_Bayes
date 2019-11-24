import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import sys
print(sys.argv)
train_data = pd.read_csv(sys.argv[1])
test_data = pd.read_csv(sys.argv[2])
train_data_length = train_data.shape[0]
X = pd.concat([train_data.review,test_data.review],axis=0).reset_index().review
Y_train = pd.get_dummies(train_data.sentiment).positive
print(train_data.info())
print(test_data.info())
def preprocess(X):
    X = X.str.lower()
    X = X.str.replace(r'.',' ')
    X = X.str.replace('[^a-zA-Z ]','')
    X = X.str.replace(r'\s+', ' ')
    X = X.str.replace(r'([a-z])\1+', r'\1')
    print("preforming stemming")
    stopwords_set = stopwords.words('english')
    stemmer = PorterStemmer()
    for i,row in enumerate(X):
        print(i,end='\r',flush=True)
        a = ''
        for word in row.split():
            if word not in stopwords_set:
                a= a+' '+(stemmer.stem(word))
        X[i] = a
    return X
X = preprocess(X)
def get_unique_keys(X):
    words = []
    for row in X:
        w = row.split()
        words.extend(w)
    return sorted(list(set(words)))    
unique_words = get_unique_keys(X[:train_data_length])
print(len(unique_words))
sum_df = pd.DataFrame(np.zeros(len(unique_words)*2).reshape(2,len(unique_words)),columns=unique_words)
for i,row in enumerate(X[:train_data_length]):
    print(i,end="\r",flush=True)
    sum_df.iloc[Y_train[i],:][list(set(row.split()))]+= 1
print("yaha tk chal gya")
unique_words_all = get_unique_keys(X)
unknown_words = list(set(unique_words_all)-set(unique_words))
df2 = pd.DataFrame(np.zeros(len(unknown_words)*2).reshape(2,len(unknown_words)),columns=unknown_words)
sum_df = pd.concat([sum_df,df2],axis=1)+1
print("yaha tk bhi chal gya")
print(sum_df.shape)
class_positive = np.sum(Y_train==1)
class_negative = np.sum(Y_train==0)
print(class_negative,class_positive)
sum_df.iloc[0,:] = np.log(sum_df.iloc[0,:]/(class_negative+1))
sum_df.iloc[1,:] = np.log(sum_df.iloc[1,:]/(class_positive+1))
Y_pred = []
for i,row in enumerate(X[train_data_length:]):
    print(i,end="\r",flush=True)
    words = list(set(row.split()))
    class_0 = np.log(class_negative/(class_positive+class_negative))+sum_df.iloc[0,:][words].sum()
    class_1 = np.log(class_positive/(class_positive+class_negative))+sum_df.iloc[1,:][words].sum()
    if(class_0<class_1):
        Y_pred.append(1)
    else:
        Y_pred.append(0)
np.savetxt(sys.argv[3],np.array(Y_pred))        
