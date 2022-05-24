# Data analysis
import os
import pandas as pd
from tqdm import tqdm

# NLP
import spacy
nlp = spacy.load("en_core_web_sm")

# Plotting
import numpy as np
import matplotlib.pyplot as plt

# sentiment analysis VADER
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
# sentiment with spacyTextBlob
from spacytextblob.spacytextblob import SpacyTextBlob
nlp.add_pipe('spacytextblob')

# visualisations
import matplotlib.pyplot as plt


def sentiment():
    
    # Get the filepath
    filepath = os.path.join("..", "..", "CDS-LANG", "tabular_examples", "fake_or_real_news.csv")
    # load the data
    data = pd.read_csv(filepath)
    
    
    # Splitting into two datasets, one of real news and one of fake 
    # First, I group the dataset by the coloumn i want to focus on which in this case is the column "label"
    fake_news = data.groupby(data.label)
    # Next, is to get the specific rows i that group which is the ones called FAKE
    fake_news_df = fake_news.get_group("FAKE")
    # Result is the new dataset of fake news
    fake_news_df
    
    # I then do the same with real news 
    real_news = data.groupby(data.label)
    real_news_df = real_news.get_group("REAL")

    
    # Using VADER to get sentiment scores for all headlines in fake news df
    # Creating a new empty list where my results can be shown
    fake_news_scores = []
    # For every headline in fake_news dataset, calculate the polarity (sentiment) score
    for headline in fake_news_df["title"]:
        dict_of_scores = analyzer.polarity_scores(headline)
        # Add the scores to the list of fake_news_scores
        fake_news_scores.append(dict_of_scores)

    # Creating a dataframe with the sentiment scores for fake_news_df
    fake_news_scores_df = pd.DataFrame(fake_news_scores, columns = ["neg", "neu", "pos", "compound"])

    
    # Using NER to find every individual occurrence of geopolitical entities
    fake_news_GPE = []
    # For every headline in the fake_news data set and for the entities within the headlines, find the entity called GPE
    for headline in tqdm(nlp.pipe(fake_news_df["title"], batch_size = 500)):
        for entity in headline.ents:
            if entity.label_ == "GPE":
                # Add the entities called GPE to the list of fake_news geopolitical mentions
                fake_news_GPE.append(entity.text)

    
    # Creating a dataframe for all of the geopolitical mentions 
    fake_news_GPE_df = pd.DataFrame(fake_news_GPE, columns = ["GPE"])

    
    # Using the same methods as for fake_news on real_news
    # Using VADER to get sentiment scores for all headlines
    real_news_scores = []
    # For every headline in real_news dataset, calculate the polarity (sentiment) score
    for headline in real_news_df["title"]:
        dict_of_scores = analyzer.polarity_scores(headline)
        # Add the scores to the list of real_news_scores
        real_news_scores.append(dict_of_scores)

    # Creating a dataframe with the sentiment scores for real_news_df
    real_news_scores_df = pd.DataFrame(real_news_scores, columns = ["neg", "neu", "pos", "compound"])

    # Using NER to find every individual occurrence of geopolitical entities
    real_news_GPE = []
    # For every headline in the fake_news data set and for the entities within the headlines, find the entity called GPE
    for headline in tqdm(nlp.pipe(real_news_df["title"], batch_size = 500)):
        for entity in headline.ents:
            if entity.label_ == "GPE":
                # Add the entities called GPE to the list of fake_news geopolitical mentions
                real_news_GPE.append(entity.text)
                
                
                
    # Creating a dataframe for all of the geopolitical mentions 
    real_news_GPE_df = pd.DataFrame(real_news_GPE, columns = ["GPE"])
    
    
    # Saving a CSV which shows the text ID, the sentiment scores, and column showing all GPEs in both fake news dataset and in real news dataset
    # Defining the columns and its inputs for the CVS
    # Defining the columns I want in my CSV by zipping them together in a list 
    list_of_columns_fn = list(zip(fake_news_df["Unnamed: 0"], fake_news_scores, fake_news_GPE))
    # Making a new dataframe with the names of the wanted columns as well as the values i want within them
    fake_news_dframe = pd.DataFrame(list_of_columns_fn, columns = ["TextID", "Sentiment Scores", "GPE Mentions"])
    # Writing that dataframe to a new csv file (see seperate .csv for results
    fake_news_dframe.to_csv("../output/Fake_news.csv", encoding = "utf-8")
    #print(fake_news_dframe)
    
    
    # Same for the real news dataset
    list_of_columns_rn = list(zip(real_news_df["Unnamed: 0"], real_news_scores, real_news_GPE))

    real_news_dframe = pd.DataFrame(list_of_columns_rn, columns = ["TextID", "Sentiment Scores", "GPE Mentions"])
    real_news_dframe.to_csv("../output/Real_news.csv", encoding = "utf-8")
    #print(real_news_dframe) 
    
    
    # FAKE NEWS DATASET

    # To find the top 20 most common entities I use the value_counts function wich counts the frequency of GPE's
    fake_news_most_common = fake_news_GPE_df['GPE'].value_counts()
    # Making it into a dataframe in order to use the function nlargest in the next step
    fake_news_most_common_df = pd.DataFrame(fake_news_most_common, columns = ["GPE"])
    # Using nlargest to find the top 20 GPE's of the most frequent list 
    top_twenty = (fake_news_most_common_df['GPE'].nlargest(n=20))
    # type(top_twenty)
    # Because this is panda.series type, I will have to make it into a list of pairs in order to plot it into a bar chart 
    # I do that using .tolist() and zipping the entities with it's value
    top_twenty_list = top_twenty.tolist()
    list_of_most_common_fn = list(zip(top_twenty.index, top_twenty))
    #print(list_of_most_common_fn)

    # REAL NEWS DATASET
    real_news_most_common = real_news_GPE_df['GPE'].value_counts()
    real_news_most_common_df = pd.DataFrame(real_news_most_common, columns = ["GPE"])
    top_twenty1 = (real_news_most_common_df['GPE'].nlargest(n=20))

    top_twenty_list1 = top_twenty1.tolist()
    list_of_most_common_rn = list(zip(top_twenty1.index, top_twenty_list1))
    #print(list_of_most_common_rn)

    
    
    # Now that I have my list of key, value pairs I can plot them into a bar chart using matplotlib

    #FAKE NEWS BAR CHART
    # First I will need to assign my list of pairs to x,y 
    # Doing that by unzipping(*) my list of paris, in order to assign key to x and the value of that key to y
    labels, y = zip(*list_of_most_common_fn)
    x = np.arange(len(labels)) 

    # Then plotting it into the bar chart
    plt.xticks(x, labels)
    plt.yticks(y)


    # Making it look pretty and trying to make up for the fact that i can't figure out how to make the y axis readable
    # Naming the x and y labels
    plt.xlabel("GPE Entities")
    plt.ylabel("No. of mentions in headlines")
    # Giving the bar chart a title
    plt.title("T0p 20 entities of the fake news dataset")
    # Costumizing the xaxis
    plt.bar(x, y, color = "pink", width = 0.8)
    plt.xticks(rotation=75)

    # The final result saved as a picture
    plt.savefig('../output/TOP 20 ENTITIES OF FAKE NEWS')
    
    
    #REAL NEWS BAR CHART
    labels, y = zip(*list_of_most_common_rn)
    x = np.arange(len(labels)) 

    plt.xticks(x, labels)
    plt.yticks(y)

    plt.xlabel("GPE Entities")
    plt.ylabel("No. of mentions in headlines")
    plt.title("T0p 20 entities of the real news dataset")
    plt.bar(x, y, color = "red", width = 0.8)
    plt.xticks(rotation=75)

    plt.savefig('../output/TOP 20 ENTITIES OF REAL NEWS')
    
    print("Script succeeded, the results can be seen in output-folder")
    

sentiment()