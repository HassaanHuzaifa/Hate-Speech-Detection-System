# import the required libraries
import pandas as pd #improting the pandas library
import numpy as np #importing the numpy library 
import re #importing the re library of python
import seaborn as sns #importing the seaborn library of python to use in the project.
import matplotlib.pyplot as plt #importing the matplotlib library to plot the graphs in projects.

from matplotlib import style #importing the style library from matplotlib
from nltk.tokenize import word_tokenize #improting teh word_tokenize function of nltk library
from nltk.stem import WordNetLemmatizer #importing the wordNetLemmatizer function from nltk.stem to lemmatize the word
from nltk.corpus import stopwords #importing the stopwords from nltk library
stop_words = set(stopwords.words('english'))

from wordcloud import WordCloud #importing wordcloud from word cloud library of python
from sklearn.feature_extraction.text import TfidfVectorizer #importing the TfidfVectorizer from sklearn.features
from sklearn.model_selection import train_test_split #importing the test_train function from sklearn to split the data into testing and training
from sklearn.linear_model import LogisticRegression #importing the Logistic Regression from sklearn.linear library
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay #importing the accuracy_score, nad classification report, confusion matrix and Confusion Matrix Display 
