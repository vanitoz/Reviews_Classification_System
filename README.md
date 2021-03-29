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

Data for this project was sourced from yelp.com. With API requests library was succesful collected information about more then 1600 cafes and bakeries in New York City. More then 41 000 business reviews sucessfuly web scraped and loaded into main Data Base.

Each review on yelp source labeled with rating provided by customer.  For this project reviews classified with rating 1 and 2 was labeled as 'Negative' class, 4 and 5 labeled as 'Positive' and reviews with class 3 labeled as 'Neutral'. Graph below on a left side explain original scraped data from web with 5 classes. Right side graph shows labeled data for this project based on original information about reviews.

<p align="center">
    <img src="images/Labels.png" alt="drawing" width="900" hight="700"/>

With appropriate functions from Pandas library some insides was found from the data. Graf below shows average length of review per each class. 'Negative' reviews on average tend to have more words then 'Positive'. This could be important variable for features engineering.

<p align="center">
    <img src="images/words_distribution.png" alt="drawing" width="700" hight="350"/>

After appropriate Pre-Processing that include Tokenization, Removing Stop-words and Cleaning Data next step was to generate frequency distribution of words within the whole corpus. It helped to understand data better and explained to us what kind of additional cleaning needs to be done before turning data into a Document-Term Matrix. Graph below shows the 20 most frequent 3-gram words that was found in each class.

<p align="center">
    <img src="images/3-gram.png" alt="drawing" width="800" hight="900"/>

All classes contain lots of the same words. Because of the similarities of each label’s vocabulary, it could be difficult for machine learning algorithms to differentiate between them and determine target variable. Graph below represents a venn diagram that shows how many unigram and 3 -gram words belong to each class and how many words show up in both classes. It helps to understand that unigram and 3 -gram  must be important variables during features engineering.

<p align="center">
    <img src="images/Venns.png" alt="drawing" width="800" hight="400"/>

## Modeling

Weighted F1 score and Accuracy was used as main evaluation metrics for this project. Starting with baseline models, Random Forest, XGBoost , LightGBM was applied to processed data. Best result was shown by XGBoost and LightGBM Model with Weighted F1-score = 88 and Acuracy - 81%
Next step was to run Gridsearch on the same 3 models. Following table shows the performance of each model with best hyperparameters found by Gridsearch.

<p align="center">
    <img src="images/Results.png" alt="drawing" width="500" hight="250"/>
 
Based on results, the highest Accuracy and weighted F-1 score achieved with LightGbm classifier with litter overfiting.
Testing Accuracy: 0.89 and weighted F1 Score: 0.79
The Confusion Matrix below explains the high True Positive rate for. In this business context, we would ideally want as many True Positives as possible, because that would be identifying Hate Speech correctly.

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
    ├── data                         # Data Base with scraped data. CSV tables with processed data.
    ├── images                       # Both sourced externally and generated from Code"       
    └── pickles                      # Results of Grid Search for each model   
    └── modules                      # notebooks and files with main functions
        |─-data_collection.ipynb
        |──data_processing.ipynb
        |──eda.ipynb
        |──modeling.ipynb
        └──utils.py
    
    
**References:**  


**Contact Information:** <br>
[Ivan Zakharchuk](https://github.com/vanitoz)<br>


