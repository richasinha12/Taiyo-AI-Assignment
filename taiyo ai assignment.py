#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from time import sleep
from numpy import tensordot
from selenium.common.exceptions import *
from click import NoSuchOption
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import pandas as pd


final_data=pd.DataFrame() #Initiating Dataframe to store data
tender_name_list=[]#Initiating List to store respective tender name
Industry_list=[]#Initiating List to store respective industry
Location_of_contract_list=[]#Initiating List to store respective location of contract
Value_of_contract_list=[]#Initiating List to store respective value of contract
Procurement_reference_list=[]#Initiating List to store respective procurment 
Published_date_list=[]#Initiating List to store respective published date
Closing_date_list=[]#Initiating List to store respective closing date
Closing_time_list=[]#Initiating List to store respective clossing time

path = r'geckodriver.exe'
driver = webdriver.Firefox(executable_path=path)

for z in range(1,7):  #Note-Increase the range and we can scrape more data
    try:
        driver.get(f'https://www.contractsfinder.service.gov.uk/Search/Results?page={z}#07c879eb-5e62-435b-8034-10e114ec9938')
        sleep(2) 
        
        for x in range(1,21):
                try:
                    driver.find_element(By.XPATH,f'//div[3]/div/div/div/div[1]/div[{x}]/div[1]/h2/a').click()
                    sleep(2)
                    try:
                        tender_name=driver.find_element(By.XPATH,'//h1[@class="govuk-heading-l break-word"]')
                        tender_name_list.append(tender_name.text)
                    except:
                        tender_name_list.append('tender name missing')

                    try:
                        industry=driver.find_element(By.XPATH,'//*[@id="content-holder-left"]/div[3]/ul/li/p')
                        Industry_list.append(industry.text)
                    except:
                        Industry_list.append('industry missing')

                    try:    
                        location_of_contract=driver.find_element(By.XPATH,'//*[@id="content-holder-left"]/div[3]/p[2]/span')
                        Location_of_contract_list.append(location_of_contract.text)
                    except:
                        Location_of_contract_list.append('location of contract missing')
                    
                    try:
                        value_of_contract=driver.find_element(By.XPATH,'//*[@id="content-holder-left"]/div[3]/p[3]')
                        Value_of_contract_list.append(value_of_contract.text)
                    except:
                        Value_of_contract_list.append('value of contract missing')

                    try:
                        procurement_reference=driver.find_element(By.XPATH,'//*[@id="content-holder-left"]/div[3]/p[4]')
                        Procurement_reference_list.append(procurement_reference.text)
                    except:
                        Procurement_reference_list.append('procurement reference missing')

                    try:
                        published_date=driver.find_element(By.XPATH,'//*[@id="content-holder-left"]/div[3]/p[5]')
                        Published_date_list.append(published_date.text)
                    except:
                        Published_date_list.append('published date missing')

                    try:
                        closing_date=driver.find_element(By.XPATH,'//*[@id="content-holder-left"]/div[3]/p[6]')
                        Closing_date_list.append(closing_date.text)
                    except:
                        Closing_date_list.append('closing date missing')
                    
                    try:
                        closing_time=driver.find_element(By.XPATH,'//*[@id="content-holder-left"]/div[3]/p[7]')
                        Closing_time_list.append(closing_time.text)
                    except:
                        Closing_time_list.append('closing time missing')

                        
                except:
                    print('NO DATA RECEIVED')
                driver.back()
                
    except:
        print('INVALID URL')


final_data['Tender name']=tender_name_list
final_data['Industry']=Industry_list
final_data['Location of contract']=Location_of_contract_list
final_data['Value of contract']=Value_of_contract_list
final_data['Procurment references']=Procurement_reference_list
final_data['Published date']=Published_date_list
final_data['closing date']=Closing_date_list
final_data['closing time']=Closing_time_list
final_data.to_csv(r"D:\amazon dataset\richa.csv")


# In[34]:


# importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[35]:


df = pd.read_csv("richa.csv")
df.head()


# In[36]:


df.shape


# In[37]:


df.info()


# In[38]:


#cleaning
df['Procurment references'] =df['Procurment references'].str.replace("Procurment references","")


# In[39]:


df['Value of contract'] =df['Value of contract'].str.replace("Value of contract","")


# In[40]:


df['Location of contract'] =df['Location of contract'].str.replace("Location of contract","")


# In[41]:


df['closing date'] =df['closing date'].str.replace("closing date","")


# In[43]:


df['closing time'] =df['closing time'].str.replace("closing time","")


# In[44]:


df['Industry'] =df['Industry'].str.replace("Industry","")


# In[45]:


df.describe()


# In[46]:


df['Location of contract'].value_counts()


# In[47]:


#contract values
newdf = df["Value of contract"].str.split(" ", n = 4, expand = True)


# In[48]:


newdf


# In[49]:


#taking highest values
newdf[2]=newdf[2].str.replace("£","")
newdf


# In[50]:


df['Value of contract'] = newdf[2]


# In[51]:


df['Value of contract'].fillna(0, inplace = True)


# In[52]:


df['Value of contract'] =df['Value of contract'].fillna(0)
df['Value of contract'] =df['Value of contract'].fillna(0)


# In[53]:


df.isnull().sum()


# In[54]:


df.head()


# In[55]:


df['closing date'] = pd.to_datetime(df['closing date'], errors='coerce')


# In[56]:


df['Published date'] = pd.to_datetime(df['Published date'], errors='coerce')


# In[57]:


df.head()


# Time series data Visualization in Python

# In[58]:


df.index = df['closing date']
del df['closing date']


# In[59]:


df.head()


# In[60]:


df.shape


# In[61]:


# deleting column
df.drop(columns='Unnamed: 0')


# In[62]:


df["Value of contract"] = df["Value of contract"].str.replace("[\$\,\.]", "")


# In[63]:


df.dtypes


# In[64]:


df["Value of contract"].astype(float).plot()


# # Plotting a simple line plot for time series data.

# In[ ]:


df['Value of contract'].plot()


# In[ ]:


#Now let’s plot all other columns using subplot.

df.plot(subplots=True, figsize=(10, 12))


# In[ ]:


# Resampling the time series data based on monthly 'M' frequency
df_month = df.resample("M").mean()

# using subplot
fig, ax = plt.subplots(figsize=(10, 6))

# plotting bar graph
ax.bar(df_month['2016':].index,
	df_month.loc['2016':, "Volume"],
	width=25, align='center')


# Differencing is used to make the difference in values of a specified interval. By default, it’s one, we can specify different values for plots. It is the most popular method to remove trends in the data.

# In[ ]:


df.Low.diff(2).plot(figsize=(10, 6))


# In[ ]:


df.High.diff(2).plot(figsize=(10, 6))


# Plotting the Changes in Data
# We can also plot the changes that occurred in data over time. There are a few ways to plot changes in data.
# 
# Shift: The shift function can be used to shift the data before or after the specified time interval. We can specify the time, and it will shift the data by one day by default. That means we will get the previous day’s data. It is helpful to see previous day data and today’s data simultaneously side by side.

# In[ ]:


df['Change'] = df.Close.div(df.Close.shift())
df['Change'].plot(figsize=(10, 8), fontsize=16)


# In this code, .div() function helps to fill up the missing data values. Actually, div() means division. If we take df. div(6) it will divide each element in df by 6. We do this to avoid the null or missing values that are created by the ‘shift()’ operation. 
# 
# Here, we have taken .div(df.Close.shift()), it will divide each value of df to df.Close.shift() to remove null values.

# We can also take a specific interval of time and plot to have a clearer look. Here we are plotting the data of only 2017.

# In[ ]:


df['2017']['Change'].plot(figsize=(10, 6))


# Time Series Box and Whisker Plots by Interval

# In[ ]:


#from pandas import read_csv
from pandas import DataFrame
from pandas import Grouper
from matplotlib import pyplot
#series = read_csv('richa.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
groups = series.groupby(Grouper(freq='A'))
years = DataFrame()
for name, group in groups:
	years[Tender name.closing date] = group.values
years.boxplot()
pyplot.show()


#  box and whisker plot is created for each month-column in the newly constructed DataFrame.

# In[ ]:


# create a boxplot of monthly data
#from pandas import read_csv
#from pandas import DataFrame
#from pandas import Grouper
#from matplotlib import pyplot
from pandas import concat
#series = read_csv('daily-minimum-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
one_year = series['1990']
groups = one_year.groupby(Grouper(freq='M'))
months = concat([DataFrame(x[1].values) for x in groups], axis=1)
months = DataFrame(months)
months.columns = range(1,13)
months.boxplot()
pyplot.show()


# # Implementation of NER Using spaCy

# Spacy is an open-source NLP library that provides various facilities and packages which can be help full on NLP tasks such as POS tagging, lemmatization, fast sentence segmentation 
# 
# Let’s get started with importing libraries.
# 
# import spacy

# In[ ]:


import spacy


# In[ ]:


raw_text="""df['Tender name']"""


# In[ ]:


#Loading only the NER model of spicy.

NER = spacy.load("en_core_web_sm", disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])


# In[ ]:


#Fitting the model on the sample text.

text= NER(raw_text)


# In[ ]:


#Printing the named entity found by the model in our sample text.

for w in text.ents:
    print(w.text,w.label_)


# # Word2Vec in Python

# In[ ]:


pip install gensim
pip install nltk


# In[ ]:


from nltk.tokenize import sent_tokenize, word_tokenize
import gensim
from gensim.models import Word2Vec


# In[ ]:


#Preparing the corpus
#We create the list of the words that our corpus has using the following lines of code:

corpus_text = 'n'.join(rev[:1000]['Text'])
data = []
# iterate through each sentence in the file
for i in sent_tokenize(corpus_text):
    temp = []
    # tokenize the sentence into words
    for j in word_tokenize(i):
        temp.append(j.lower())
    data.append(temp)


# In[ ]:


#Building the Word2Vec model using Gensim
#To create the word embeddings using CBOW architecture or Skip Gram architecture, you can use the following respective lines of code:

model1 = gensim.models.Word2Vec(data, min_count = 1,size = 100, window = 5, sg=0) 
model2 = gensim.models.Word2Vec(data, min_count = 1, size = 100, window = 5, sg = 1)

