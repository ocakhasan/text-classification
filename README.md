# text-classification

[Ekşi Sözlük](https://eksisozluk.com/) is one of the most popular social media platforms in Turkey. Users discuss every topic such as (Education, Relationships etc). 
So I wanted to create a simple text classification example. 

## DATASET

I created dataset by scraping the website. You can use the **scrape.py** to do it.
It will scrape the topics currently popular (agenda). It will save entries to the *total_dataset.csv* file. 

For the project, I uploaded a small dataset where there are 8000 entries, 1600 for 5 categories.

### LABELING DATA
Since text classification is a supervised task, you need to label the data. 
I labeled as Economy, Education, Politics, Relationships, Sports. 

But you can choose as many as categories you want.

## CREATING DATASET

You need to run **data.py** to get the dataset which you can use for training. 
From the command line, just write
```
python data.py
```
It will create a folder named dataset, and it will include the features and labels for both  training and testing. 

To create the dataset, Tf-Idf is used. You can check from [here](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)

## TRAINING

I just used the [Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html). 
I got 77 percent accuracy which is not bad actually. 
You need to run the **train.py** file.
From the command line, just write
```
python train.py
```
It will create a folder named as models, where you can find your Naive Bayes model used for training.

## TESTING
To test some tweets, you need to run **test.py**
I wrote 3 tweet just for trying, you can challenge the model as many as you can.
Do not forget, the training data is Turkish, this is a Turkish text classification so you need to write your sentences in Turkish for testing.
From the command line, just write
```
python test.py
```



