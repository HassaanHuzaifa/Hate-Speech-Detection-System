# Hate Speech Detection System

The term “hate speech” refers to any speech that disparages an individual or a group based on one or more characteristics, including race, color, ethnicity, gender, sexual orientation, nationality, religion, or another feature. The volume of hate speech is also steadily rising due to the enormous growth of user-generated web content, particularly on social media networks.

As basic word filters do not adequately address this issue, it is necessary to do natural language processing that particularly targets it. The domain of a statement, its discourse context, as well as context made up of co-occurring media objects (such as images, videos, and audio), the precise time of posting and current global events, the identity of the author, and the identity of the targeted recipient, are the things that can have an impact on what is deemed to be hate speech.


Logistic Regression
Logistic regression is a statistical model which can be used in this project. The algorithm’s output creates a probability value that may be transferred to two or more discrete classes.

This is how the logistic regression model is expressed in formal terms.

log p(x) 1 − p(x) = β0 + x · β

When p is solved, this results

p(x; b, w) = eβ0+x·β 1 + eβ0+x·β = 1 1 + e−(β0+x·β)

You’ll notice that understanding the overall specification in terms of the transformed probability is much simpler than understanding it in terms of the untransformed probability.

We should forecast Y = 1 when p ≥ 0.5 and Y = 0 when p < 0.5. to reduce the misclassification rate. This entails predicting 0 or 1 depending on whether β0 + x ·β is positive or negative. Consequently, logistic regression provides a linear classifier.

The answer to the equation ( β0 + x · β = 0 ) is a decision boundary that separates the two predicted classes. If x is one-dimensional, a point will be the solution; if x is two-dimensional, a line will be the solution, etc. One may demonstrate (do the exercise!) that separate from the choice border. The class probabilities rely on distance from the boundary in a certain way, and they tend to go towards the extremes (0 and 1), according to logistic regression, which also identifies the location of the class. These statements about probabilities make logistic regression more than just a classifier. It makes stronger, more detailed predictions and can be fit differently, but those strong predictions could be wrong.

Regression analysis that is utilized when the dependent variable is binary is called logistic regression. As with SVC and MNB, we trained LR in the same manner. Utilizing L2 regularisation, the hyper-parameter C was left at its default value of 1.0. Using TF-IDF weights, combined word and character n-gram features provide greater accuracy for both languages.

TFIDF
TF-IDF is a combination of two words, i.e. Term Frequency and Inverse Document Frequency. First, the term “term frequency” will be discussed. TF is used to determine how frequently a term appears in a document. Consider a paper called “T1” that has 5000 words in it and exactly 10 instances of the term “Alpha.” Since it is commonly known that papers can range in size from extremely brief to quite lengthy, it is possible that any term may appear more frequently in lengthy documents than in shorter ones. In order to solve this problem, the word frequency is calculated by dividing each instance of a term in a document by the total number of terms in that document. The term frequency of the word “Alpha” in the document “T1” will therefore be TF = 10/5000 = 0.002 in this instance.

The two key metrics that show the specificity and relevance of words with the information carried by the documents are term frequency (TF) and inverse document term frequency (IDF). For the n-gram features that we retrieved from the tweets in the datasets, we utilized TF-IDF weights. Our model employs word n-grams of order for the English language (1, 2). 38536 features were retrieved with this. The same 81191 attributes were obtained via character n-grams of order (1, 5). Combining character n-grams (1, 5) and word n-grams (1, 2) yielded 119727 characteristics.

The three feature extraction techniques were tested in order to determine which produced the best feature model. The three feature models mentioned above were all used in the case of Tamil. 117173 characteristics were retrieved from the word n-grams of the (1, 4) order. Word n-grams (1, 4) and character n-grams (1, 7) together have retrieved 443075 features, while the character n-grams of order (1, 7) have extracted 325902 characteristics. These n-grams can be used to identify localized, minute syntactic patterns in text using flexible language. A sample of these traits in Malayalam and Tamil language text found in the relevant datasets are shown in Tables 2 and 3, respectively.

Project Prerequisites
The requirement for this project is Python 3.6 installed on your computer. I have used Jupyter notebook for this project. You can use whatever you want.
The required modules for this project are –

Numpy(1.22.4) – pip install numpy
Seaborn(0.9.0) – pip install seaborn
Keras(2.9.0) – pip install keras
Pandas(1.5.0) – pip install pandas
That’s all we need for our project.
