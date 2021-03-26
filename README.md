# Reviews Classification System With NLP

<p align="center">
    <img src="images/customer-review-Inner-service-banner.png" alt="drawing" width="1000" hight="400"/>

## Overview
Main goal of this project is to create a model that will be able to classify customers reviews using machine learning multi classification algorithms.
Random Forest, XGBoost, LightGbm algorithms used to create baseline models. 
The final model was a XGBoost that used TfidfVectorizer for feature engineering. It produced an F1 of 0.8 and Accuracy of .



## Business Problem

   With the ever expanding world we are privy to using apps such as Yelp to discover new bars, restaurants, cafes and services, there is a sense of being overwhelmed by the sheer number of options. We use ratings and reviews to better get a better idea of a potential trip, but oftentimes the ratings that people leave are divorced from the depth of the reviews they post. For example someone might leave a 4 star rating, but their review belies a “true” rating closer to a 3. This is what my project hopes to develop, using Natural Language Processing, is projecting a ‘true’ rating, alongside the actual, that will better classify the quality of a store. Here I am fixing the disconnect between the reviews people write out, and the ratings they leave.


## Approach

General Approach for this problem was based on Cross Industry Standard Process for Data Mining (CRISP-DM)
Which includes all following impotrtant steps: 

1. Look at the big picture. 
2. Get the data. 
3. Discover and visualize the data to gain insights. 
4. Prepare the data for Machine Learning algorithms. 
5. Select a model and train it. 
6. Fine-tune your model. 
7. Present your solution. 
8. Launch, monitor, and maintain system.

<p align="center">
    <img src="images/approach.png" alt="drawing" width="500" hight="200"/>
    
## Methodology

Based on our business problem we are trying to accomplish certain tasks that involve natural language. NLP allows computers to interact with text data in a structured and sensible way. With NLP, computers are taught to understand human language, its meaning and sentiments. In order to translate complex natural human language into systematic constructed features we need to follow some major steps which showed on the next graph.

<p align="center">
    <img src="images/NLP_protcess.png" alt="drawing" width="600" hight="300"/>

## Analysis

Data for this project was sourced from yelp.com. With API requests library was succesful collected information about more then 1600 cafes and bakeries in New York City. More then 41 000 business reviews sucessfuly web scraped and inserted to main Data Base.

Each review on yelp source labeled with rating provided by customer.  For this project reviews classified with rating 1 and 2 was labeled as 'Negative' class, 4 and 5 labeled as 'Positive' and reviews with class 3 labeled as 'Neutral'. Graph below on a left side explain original scraped data from web with 5 classes. Next to it graph shows labeled date for this project based on original information about reviews 

 <p float="center">
  <img width="450" height="350" src=images/5_class.png>
  <img width="450" height="350" src=images/3_class.png>
 </p>
 
After appropriate Pre-Processing that include Tokenization, Removing Stop-words and Cleaning Data we were able to generate frequency distribution of words within the whole corpus. It helped to understand data better and explained to us what kind of additional cleaning needs to be done before turning data into a Document-Term Matrix. Graph below shows the 25 most frequent words that we were able to find in each class that belong to the main corpus.

<p align="center">
    <img src="images/words_distribution.png" alt="drawing" width="700" hight="350"/>

With the WordCloud library we were able to create bags of most important words in each class. We also observed that both classes had lots of the same words that were located in the corpus of our data. Because of the similarities of each label’s vocabulary, it could be difficult for machine learning algorithms to differentiate between them and determine what counts as hate speech.

<p align="center">
    <img src="images/3-gram.png" alt="drawing" width="800" hight="900"/>

With further analysis we were able to find out and create vocabulary of only words that belong to tweets labeled as hate speech. We found 6312 words that exclusively belong to tweets labeled as hate speech. Majority of hate speech words are racist, sexist and homophobic slurs that exceed cultural slang. The fact that these words are unique to the "Hate Speech" label affirm that it's indeed hate speech that should be flagged and taken down.


<p align="center">
    <img src="images/Venns.png" alt="drawing" width="700" hight="350"/>

Graph above represents a venn diagram that shows how many unique words belong to each class and how many words show up in both classes. 3391 words showing up in both classes which makes it difficult for machine learning models to predict the correct label on particular tweets.
After futers engineering with TF-IDF Vectorization the next step takes place for creating models and evaluating them.

## Modeling

F1 score and Recall was used as main evaluation metrics for this project. We want to classify correct hate speech as much as possible and so that it can be efficiently removed. 
Starting with baseline models, Random Forest, Naive Bayes, Logistic Regression was applied to imbalanced data. Best result was shown by Random Forest Model with Recall = 12% and F1-score = 19%

Next step was to run the same 3 models on balanced data. Following table shows the performance of each model on the test set.

<p align="center">
    <img src="images/results.png" alt="drawing" width="500" hight="250"/>
 
Based on results, the highest Recall and F-1 score achieved with Random Forest and Naive Bayes classifier. Following step was to use GridSearch with a Random Forest classifier to get the best parameters in order to achieve a higher Recall score. Random Forest with Hyper Parameters selected with GridSearch let us create final model with following results on testing data: 
Precision: 0.7124
Recall: 0.937
Testing Accuracy: 0.7816
F1 Score: 0.8094


The Confusion Matrix below explains the high True Positive rate. In this business context, we would ideally want as many True Positives as possible, because that would be identifying Hate Speech correctly.

<p align="center">
    <img src="images/conf_mtrx.png" alt="drawing" width="900" hight="500"/>

## Conclusion
The final model performance was achieved through balancing data with additional tweets labeled as hate speech. 
The biggest part of the project has been done with Exploratory Data Analyses. It showed specific insides of the data. 
Final model was created with Random Forest Classifier and selection of the best parameters from GridSearch.
Hate speech detection is an extremely difficult task for Machine Learning because of the nuances in English slang and slurs. 
This project shows that we were able to create a system that can provide content moderation with pretty good results.


## Future Work

One of the further steps will be to expand this project for multi classification problems. For example, we can classify other types of tweets like offensive speech. Also we can evaluate model with new tweets or other online forum data to see if it can generalize well. 


## Repository Structure

    ├── README.md                    # The top-level README for reviewers of this project"
    ├── data                         # Synergized Data obtained from University of Michigan and Detroit Open Data Portal"
    ├── modules                      # py files with functions for ingeniring and modeling"
    ├── images                       # Both sourced externally and generated from Code"       
    ├── modeling.ipynb               # Notebook that gpes pver out modling process"                                        
    └── features_engineering.ipynb    # Notebook Used for feature engineering before Modeling"
    
    
**References:**  


**Contact Information:** <br>
[Ivan Zakharchuk](https://github.com/vanitoz)<br>


