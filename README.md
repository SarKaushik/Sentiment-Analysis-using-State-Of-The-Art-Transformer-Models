I leverage AWS services to conduct sentiment analysis on customer reviews using transformer models. Customer reviews are stored in an Amazon S3 bucket, and the heavy computational tasks are executed on a GPU-enabled Amazon EC2 instance. The EC2 instance runs advanced transformer models like BERT and RoBERTa to analyze the sentiment in the feedback. This architecture ensures efficient and scalable processing of large datasets, optimizing performance with AWS’s cloud infrastructure.



##AWS Architecture

![image](https://github.com/user-attachments/assets/ee91013d-abd9-4e4b-bb7f-a1a606c63d06)





## Provisioning EC2 Instance

<img width="1440" alt="EC2 Instance" src="https://github.com/user-attachments/assets/310fad44-54f3-4013-ac8c-23855288c8c8">




## Provisioning S3 Bucket

<img width="1440" alt="EC2 Instance" src="https://github.com/user-attachments/assets/cac3c8cd-c7a4-474d-b764-578e4abbbe7a">


##Indentify Customer file in S3
<img width="1240" alt="Screenshot 2024-10-06 at 3 57 13 PM" src="https://github.com/user-attachments/assets/546efd1e-9343-4bae-bc95-fc19976401b4">


#Using nano to create python script to build logic for data conversion in Python data frame


```
nano load_reviews.py


import boto3
import pandas as pd
import io

# Define S3 bucket and file details
bucket_name = 'customerreview'
file_key = 'customerreview/customer_reviews.xlsx'

# Initialize S3 client
s3 = boto3.client('s3')

# Get the Excel file from S3
response = s3.get_object(Bucket=bucket_name, Key=file_key)

# Read the Excel content into a pandas DataFrame
file_content = response['Body'].read()
excel_data = pd.read_excel(io.BytesIO(file_content))

# Display the DataFrame
print(excel_data.head())

```


##Run the script in cloudshell
'''
python3 load_reviews.py
'''







# Sentiment-Analysis-using-Transformer-LLM-s
Using sentiment analysis model to perform sentiment analysis on customer reviews for Manufacturing Industry.

Sentiment analysis using transformer models is all about teaching machines to understand and evaluate the emotional tone behind a piece of text. These models, like BERT or RoBERTa, are designed to analyze language in a more nuanced way by considering the context of each word in a sentence. Instead of just looking at individual words, transformers process the entire sentence, understanding the relationship between words, which makes them very effective at detecting whether the sentiment is positive, negative, or neutral





## Installing dependancies`

```
import locale
locale.getpreferredencoding = lambda: "UTF-8"



!pip install -q transformers


from transformers import pipeline
import pandas as pd


pd.set_option('display.max_colwidth', None)

````


Sample Review data from Customers,

```
sample_reviews = [
    "The FreshBlend juicer is a complete letdown. It barely extracts juice and leaves so much pulp behind. I’m really disappointed and returning it soon.",
    "The QuickGrow soil enhancer worked wonders in my garden! After using it for just two weeks, my flowers are blooming like never before. Highly recommend it to fellow gardeners.",
    "Bought the PowerMax Pro gaming console last month, and it's been a fantastic addition. The performance is seamless, and the game loading times are minimal. Totally worth the price!",
    "The BrightLux desk lamp seemed like a good purchase, but it's far too dim for reading. Definitely not living up to its 'ultra-bright' claim. Would not recommend."
]


```


##Converted into dataframe using Python Pandas library
```
sample_review_df = pd.DataFrame(sample_reviews, columns=['review'])
sample_review_df
```

##Loading sentiment analysis transformenr model

```
sentiment_model = pipeline('sentiment-analysis', device=0)


```

## Perform sentiment analysis for collected reviews
```
reviews = sample_review_df['review'].values
reviews

sentiment_model(reviews[0])
```

##Output
[{'label': 'NEGATIVE', 'score': 0.9995445609092712}]

##Review sentiments of all the collected customer feedbacks

```
sentiments = []

for review in reviews:
  sentiments.append(sentiment_model(review)[0]['label'])

sentiments

```


## Output 


['NEGATIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE']




An emotion detector using transformer models is designed to identify specific emotions expressed in text, such as joy, sadness, anger, or fear. These models, like BERT or GPT-based architectures, understand the context and subtleties of language, making them capable of picking up on the emotional nuances that simple keyword-based methods might miss.


##Load emotion detector trasnformer model
```
emotion_model = pipeline('sentiment-analysis',
                         model='SamLowe/roberta-base-go_emotions',
                         device=0)

```


```
sample_review_df['emotion'] = emotions
sample_review_df

```

```
emotions = []

for review in reviews:
  emotions.append(emotion_model(review)[0]['label'])

emotions
```



##Plotting some basic visuals

```
sample_review_df['sentiment'].value_counts().plot(kind='bar');
sample_review_df['emotion'].value_counts().plot(kind='bar');

```





# Question Answering Transformer Model
##A Question Answering (QA) Transformer model is designed to provide accurate answers to questions based on a given context or passage. These models, like BERT or T5, use the transformer architecture to understand the context of both the question and the reference text. The model processes the input question and the passage simultaneously, learning to locate and extract the relevant information from the passage that answers the question.


 
## Load QA Transformer Model

```
table_qa = pipeline('table-question-answering',
                    model='neulab/omnitab-large-finetuned-wtq',
                    device=0)
```


## Sample data set
```data = {
    "Year": [2020, 2016, 2012, 2008, 2004, 2000, 1996, 1992, 1988, 1984],
    "Host City": ["Tokyo", "Rio de Janeiro", "London", "Beijing", "Athens", "Sydney", "Atlanta", "Barcelona", "Seoul", "Los Angeles"],
    "Host Country": ["Japan", "Brazil", "United Kingdom", "China", "Greece", "Australia", "USA", "Spain", "South Korea", "USA"],
    "Participating Nations": [205, 207, 204, 204, 201, 199, 197, 169, 159, 140],
    "Athletes": [11338, 11238, 10568, 10942, 10625, 10651, 10318, 9356, 8391, 6829],
    "Sports": [33, 28, 26, 28, 28, 28, 26, 25, 23, 21],
    "Leading Country": ["USA", "USA", "USA", "China", "USA", "USA", "USA", "Unified Team", "Soviet Union", "USA"],
    "Total Medals": [113, 121, 104, 100, 103, 97, 101, 112, 132, 174]
}

# Creating the dataframe
table = pd.DataFrame(data)
table
```


##Define questions list
```questions = [
    "Which country won the most medals in the table for olympics?",
    "Leading country with the most medals in olympics 1988?",
    "What is the max value for Participating Nations in the olympics?",
    "Which year had the maximum Participating Nations in the olympics?",
    "Host city and host country of 2020 olympics?",
    "Host country who hosted the most number of olympic games?",
    "How many olympic games were hosted by USA?"
]
```

## Calling QA transformer model function

```
idx =[]
solutions = []
for index,qa in enumerate(questions):
    idx.append(table_qa(table, query=qa))
    solution = idx[index]
    print(qa, solution['answer'])
    #solution = idx[index]
    #print(f"Solution: {idx[index]}, Question: {qa}")
```


##Output
Which country won the most medals in the table for olympics?  USA
Leading country with the most medals in olympics 1988?  Soviet Union
What is the max value for Participating Nations in the olympics?  207
Which year had the maximum Participating Nations in the olympics?  2016
Host city and host country of 2020 olympics?  Tokyo, Japan
Host country who hosted the most number of olympic games?  USA
How many olympic games were hosted by USA?  2
























