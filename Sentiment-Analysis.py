import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from bs4 import BeautifulSoup
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

import requests
import re

# Download model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

# Collecting reviews using BeautifulSoup
pageNum = 0
start = 0
reviews = []

while True:
  req = requests.get(f'https://www.yelp.com/biz/social-brew-cafe-pyrmont?start={start}')
  soup = BeautifulSoup(req.text, 'html.parser')
  results = soup.find_all('p', {'class':"comment__09f24__gu0rG css-qgunke"})

  for i in range(len(results)):
    reviews.append(results[i].text)

  pageLimit = soup.find('span', {"class":"css-1fdy0l5"}).text
  pageLimit = int(re.findall('\d+', pageLimit)[0])
  if pageNum > round(pageLimit/10):
    break 

  start+=10
  pageNum+=1
  
df = pd.DataFrame(np.array(reviews), columns=['review'])
print(df['review'].iloc[0])
  
def sentiment_score(review):
  """
  Getting score between 1-5
  """
  tokens = tokenizer.encode(review, return_tensors='pt')
  result = model(tokens)
  return int(torch.argmax(result.logits))+1
    
print(sentiment_score(df['review'].iloc[1]))

df['sentiment'] = df['review'].apply(lambda x: sentiment_score(x[:512]))
print(df)

sns.countplot(df.sentiment)
plt.xlabel('Reviews', color = 'red')
plt.ylabel('Count', color = 'red')
plt.xticks([0, 1, 2, 3, 4],['1-Star', '2-Star', '3-Star', '4-Star', '5-Star'])
plt.title('SENTIMENT COUNT', color = 'r')
plt.show()
