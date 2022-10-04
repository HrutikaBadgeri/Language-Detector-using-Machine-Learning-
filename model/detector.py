import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore")
import pickle


# Loading the dataset
data = pd.read_csv("model/Language Detection.csv")


# value count for each language
data["Language"].value_counts()


# separating the independent and dependant features
X = data["Text"]
y = data["Language"]


# converting categorical variables to numerical
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
label_encoder = le.fit(y)
y = label_encoder.transform(y)


# creating a list for appending the preprocessed text
data_list = []


# iterating through all the text
for text in X:
    # removing the symbols and numbers
    text = re.sub(r'[!@#$(),n"%^*?:;~`0-9]', ' ', text)
    text = re.sub(r'[[]]', ' ', text)
    # converting the text to lower case
    text = text.lower()
    # appending to data_list
    data_list.append(text)


# creating bag of words using countvectorizer
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
c_vectoriser = cv.fit(data_list)
X = c_vectoriser.transform(data_list).toarray()


#train test splitting
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)


#model creation and prediction
from sklearn.naive_bayes import MultinomialNB
language_model = MultinomialNB()
language_model.fit(x_train, y_train)


# # prediction 
# y_pred = model.predict(x_test)


# # model evaluation
# from sklearn.metrics import accuracy_score, confusion_matrix
# ac = accuracy_score(y_test, y_pred)
# cm = confusion_matrix(y_test, y_pred)


# visualising the confusion matrix
# plt.figure(figsize=(15,10))
# sns.heatmap(cm, annot = True)
# plt.show()

# function for predicting language
# def predict(text):
#     x = cv.transform([text]).toarray()
#     lang = language_model.predict(x)
#     lang = le.inverse_transform(lang)
#     print("The langauge is in",lang[0])



# Saving model to disk
pickle.dump(language_model, open('LanguageDetection_model.pkl', 'wb'))
pickle.dump(c_vectoriser, open('CountVectoriser.pkl', 'wb'))
pickle.dump(label_encoder, open('LabelEncoder.pkl', 'wb'))
# # Loading model to compare the results (read)
# lang_model = pickle.load(open('model/LanguageDetection_model.pkl', 'rb'))



