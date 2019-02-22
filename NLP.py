
import pandas as pd

dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
#Tab separated values
#quoting removes quotes

import re
#nltk.download('stopwords') #To download stopwords i.e. unnecessary words like am, this, very etc
from nltk.corpus import stopwords

#Stemming i.e. finding root word of a word. e.g. love for loved
from nltk.stem.porter import PorterStemmer

corpus = []
for i in range(1000):
    review = re.sub('[^a-zA-Z]',' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review  = ' '.join(review)
    corpus.append(review)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

#Splitting of data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
from sklearn import svm
classifier = svm.SVC()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

## Feature Scaling
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)
## Fitting Logistic Regression to the Training set
#from sklearn.linear_model import LogisticRegression
#classifier = LogisticRegression(random_state = 0)
#classifier.fit(X_train, y_train)
## Predicting the Test set results
#y_pred = classifier.predict(X_test)


# =============================================================================
# #Fitting the Dataset in the Model
# from sklearn.naive_bayes import GaussianNB
# classifier = GaussianNB()
# classifier.fit(X_train, y_train)
# 
# #Predicting the values
# y_pred = classifier.predict(X_test)
# =============================================================================

#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

accuracy = ((cm[0][0]+cm[1][1])/cm.sum())*100
print('Accuracy:',accuracy)
