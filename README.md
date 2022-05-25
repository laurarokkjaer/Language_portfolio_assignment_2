
# Language Analytics - Spring 2022
# Portfolio Assignment 2

This repository contains the code and descriptions from the second assigned project of the Spring 2022 module Language Analytics as part of the bachelor's tilvalg in Cultural Data Science at Aarhus University - whereas the overall Language Analytics portfolio (zip-file) consist of 5 projects, 4 class assignments + 1 self-assigned.

## Repo structure
### This repository has the following directory structure:

| **Folder** | **Description** |
| ----------- | ----------- |
| ```input``` | Contains the input data (will be empty) |
| ```output``` | Contains the results (outputs like plots or reports)  |
| ```src``` | Contains code for assignment 2 |
| ```utils``` | Contains utility functions written by [Ross](https://pure.au.dk/portal/en/persons/ross-deans-kristensenmclachlan(29ad140e-0785-4e07-bdc1-8af12f15856c).html), and which have been used in the assignments |

Also containing a ```MITLICENSE``` for guidelines of how to reproduce and use the data in this repository, as well as a ```.txt``` reqirements-file, where the required installments will be listed.

## Assignment description
The official description of the assignment from github/brightspace: [assignment description](https://github.com/CDS-AU-DK/cds-language/blob/main/assignments/assignment2.md).

For this assignment, you will write a small Python program to perform NER and sentiment analysis using the techniques you saw in class. 

Using the corpus of Fake vs Real news, write some code which does the following:

- Split the data into two datasets - one of Fake news and one of Real news
- For every headline
- Get the sentiment scores
- Find all mentions of geopolitical entites
- Save a CSV which shows the text ID, the sentiment scores, and column showing all GPEs in that text
- Find the 20 most common geopolitical entities mentioned across each dataset - plot the results as a bar charts


### The goal of the assignment 
The goal of this assignment was to demonstrate that I have a good understanding of how to perform dictionary-based sentiment analysis
It also demonstrates that I can use off-the-shelf NLP frameworks like spaCy to perform named entity recognition and extraction.

### Data source
The data used in this assignment is the in class folder from UCloud (shared-drive/CDS-VIS/tabular_examples/fake_or_real_news.csv). 


## Methods
To solve this assignment i have worked with data analysis tool such as ```os```, ```pandas``` and ```tqdm``` (for counting). For NLP i used ```numpy``` and ```spacy.load("en_core_web_sm")```). For sentiment analysis i used ```VADER```from ```SentimentIntensityAnalyzer```. At last for visualization i used ```matplotlib```.

## Usage (reproducing results)
These are the steps you will need to follow in order to get the script running and working:
- load the given data into ```input```
- make sure to install and import all necessities from ```requirements.txt``` 
- change your current working directory to the folder above src in order to get access to the input, output and utils folder as well 
- the following should be written in the command line:

      - cd src (changing the directory to the src folder in order to run the script)
      
      - python sentiment_and_NER.py (calling the function within the script)
      
- when processed, there will be a messagge saying that the script has succeeded and that the outputs can be seen in the output folder 



## Discussion of results
The result of this script is two tables (.csv) which shows text.id of the news description, the sentiment scores for each news and the GPE (geopolitics entities) mentions for each news as well. A bar chart is also created to support the GPE, which shows the top 20 Geopolitical entities and their number of mentions both for fake and real news (two seperately charts). 

