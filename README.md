# Twitter-Sentiment-Analysis-Dash
Using Logistic Regression to classify tweets as positive and negative

*I will be demonstrating a serial way of performing twitter sentiment analysis (training and test file is present in the zip)*


1>*Let's start with sbs-preprocessing.ipynb*

```python
"""import the training dataset,emoticons are already removed,if not you can use regex to remove the emoticons"""
df = pd.read_csv("training.1600000.processed.noemoticon.csv",encoding='latin',header=None, names=cols)

# On making the boxplot we can see that some sentences(pre-cleaned) are of length greater than 140(max character
# limit for twitter comments),it means that some of the texts still contain raw html(we need to remove it).
fig, ax = plt.subplots(figsize=(5, 5))
plt.boxplot(df.pre_clean_len)
plt.show()

# Function for cleaning the text
def tweet_cleaner_updated(text):

# Storing the clean text in clean_tweet.csv

clean_df = pd.DataFrame(clean_tweet_texts,columns=['text'])
clean_df['target'] = df.sentiment
clean_df.to_csv('clean_tweet.csv',encoding='utf-8')
```
-------------------------------------------------------------------------------------------------------------------
2>*Now we will use part-2.ipynb*
*The cleaned texts will be used to train the model and create several models(models will be stored in .pkl format)*

```python
# load the csv in my_df
csv = "clean_tweet.csv"
my_df = pd.read_csv(csv,index_col=0)
#drop the rows with null text and reset the index
my_df.dropna(inplace=True)
my_df.reset_index(drop=True,inplace=True)
#Target Variable or label
test = my_df['target']
#Text
my_df.text
#convert each word into a feature
from sklearn.feature_extraction.text import CountVectorizer
cvec = CountVectorizer()
cvec.fit(my_df.text)
# total features=264939

#now we will calculate the term frequency
neg_doc_matrix = cvec.transform(my_df[my_df.target == 0].text)
pos_doc_matrix = cvec.transform(my_df[my_df.target == 4].text)
neg_tf = np.sum(neg_doc_matrix,axis=0)
pos_tf = np.sum(pos_doc_matrix,axis=0)
neg = np.squeeze(np.asarray(neg_tf))
pos = np.squeeze(np.asarray(pos_tf))
term_freq_df = pd.DataFrame([neg,pos],columns=cvec.get_feature_names()).transpose()

#storing the term frequency dataframe
term_freq_df.to_csv('term_freq_df.csv',encoding='utf-8')
```
*Now we will make several plots to analyse the data .Best visualization will be provided by neg_normcdf_hmean vs pos_normcdf_hmean .Bokeh can be used for better visualization*

**Split the dataset into train,test and validation**
*Train set has total 1564120 entries with 50.02% negative, 49.98% positive
Validation set has total 15960 entries with 49.45% negative, 50.55% positive
Test set has total 15961 entries with 49.68% negative, 50.32% positive*

*first we used python package called textblob which gave 61.84% accuracy*

*Now we will make pipeline(A pipeline is a aggregation of transformer and predictor)* 
*(TO know more about pipelines I suggest you to go to link: <http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)>*

```python
checker_pipeline = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', classifier)
        ])

# transformer used is CountVectorizer
# classifier used is Logistic Regression

 ```
 ```python
 joblib.dump(pipeline,"bigram_prat.pkl") """this function inside the function accuracy_summary will generate several models like trigram_prat.pkl,prat.pkl,better_prat.pkl"""
 ```
 
**Details of building the  model can be found in the part-2.ipynb**
-----------------------------------------------------------------------------------------------------------------
3>*Now we will use testing.ipynb*
```python
loaded_model = joblib.load("trigram_prat.pkl") """load the model"""
my_df=df.drop(['id','sentiment','date','query_string','user'],axis=1)"""dropping unwanted rows"""

y_pred = loaded_model.predict(my_df['text']) """predicting the result"""
test                                         """actual result"""
#Now we  will obtain the accurcy
from sklearn.metrics import accuracy_score 
accuracy = accuracy_score(test, y_pred)
```

*accuracy score: 80.72% (with prat.pkl)*
*accuracy score: 82.33% (with trigram_prat.pkl)*
*accuracy score: 82.53% (with better_prat.pkl)*

*Best Model:better_prat.pkl*
------------------------------------------------------------------------------------------------------------------
4>*Now we will use colleague_tester.ipynb*

*82.60% accuracy obtained on statements collected from colleagues*

-------------------------------------------------------------------------------------------------------------------
5>*Scope of improvement*

*The model is not good with sarcasms*
*The model also works with a fixed vocabulary(it has its own limitations)*



**Model Explanation**
-In this model each word represents a node in input layer
-The weights are the occurences of the word(in a normalized fashion)
-The logistic node contains a threshold which gives 1/0 (depending on the value of net*(w1x1+w2x2+-------+wnxn)*)

**Models Possible(9)**
Unigram-Without stopwords(used by me)
	With stopwords(used by me)
	Without custom stopwords(used by me)
	
Bigram-Without stopwords
	With stopwords
	Without custom stopwords
	
Trigram-Without stopwords
	With stopwords(misspelled by me as bigram_prat.pkl in part-2.ipynb)
	Without custom stopwords


									--------**THANK YOU**-------

