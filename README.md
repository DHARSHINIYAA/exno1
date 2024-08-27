# Exno:1
Data Cleaning Process

# AIM
To read the given data and perform data cleaning and save the cleaned data to a file.

# Explanation
Data cleaning is the process of preparing data for analysis by removing or modifying data that is incorrect ,incompleted , irrelevant , duplicated or improperly formatted. Data cleaning is not simply about erasing data ,but rather finding a way to maximize datasets accuracy without necessarily deleting the information.

# Algorithm
STEP 1: Read the given Data

STEP 2: Get the information about the data

STEP 3: Remove the null values from the data

STEP 4: Save the Clean data to the file

STEP 5: Remove outliers using IQR

STEP 6: Use zscore of to remove outliers

# Coding and Output 
import pandas as pd
data={'name':['John','Alice','Bob'],'age':[25,23,22],'gender':['M','M','F'],'salary':[500000,100000,700000]}
df=pd.DataFrame(data)
print(df.describe())

import pandas as pd
data={'name':['John','Alice','Bob'],'age':[25,23,22],'gender':['M','M','F'],'salary':[500000,100000,700000]}
df=pd.DataFrame(data)
grouped=df.groupby('gender').count()
print(grouped)

import pandas as pd
data={'name':['John','Alice','Bob'],'age':[25,'nil',22],'gender':['M','M','nil'],'salary':[500000,100000,700000]}
df=pd.DataFrame(data)
print(df)

import pandas as pd
data={'name':['John','Alice','Bob'],'age':[25,23,None],'gender':['M','M','F'],'salary':[500000,100000,700000]}
df=pd.DataFrame(data)
df_cleaned =df.dropna(subset=['age'])#removes duplicate
print(df_cleaned)


df_cleaned_all=df.dropna(how='all')# all values r missing in a row then it will be deleted
print(df_cleaned_all)

df_cleaned_all=df.dropna(how='any')#if any 1 value is missing then row will b deleted
print(df_cleaned_all)

df_cleaned_all=df.dropna(how='any',axis=0)
print(df_cleaned_all)

df_cleaned_all=df.dropna(how='any',axis=1)
print(df_cleaned_all)

import pandas as pd
import numpy as np
data={'name':['John','Alice','Bob','charlie','Dave','Eve','Bob','John'],
      'age':[25,23,22,np.nan,np.nan,26,22,np.nan],
      'gender':['M','M','F',np.nan,'F',np.nan,'F','F'],
      'salary':[500000,100000,700000,np.nan,10000,20000,700000,450000]}
df=pd.DataFrame(data)
print(df)
~df.duplicated()#shows the duplicate


import pandas as pd
import numpy as np
data={'name':['John','Alice','Bob','charlie','Dave','Eve','Bob','John'],
      'age':[25,23,22,np.nan,np.nan,26,22,np.nan],
      'gender':['M','M','F',np.nan,'F',np.nan,'F','F'],
      'salary':[500000,100000,700000,np.nan,10000,20000,700000,450000]}
df=pd.DataFrame(data)
df_filled=df.fillna(0)#replace the missing value with 0
print(df_filled)

import pandas as pd
import numpy as np
data={'name':['John','Alice','Bob','charlie','Dave','Eve','Bob','John'],
      'age':[25,23,22,np.nan,np.nan,26,22,np.nan],
      'gender':['M','M','F',np.nan,'F',np.nan,'F','F'],
      'salary':[500000,100000,700000,np.nan,10000,20000,700000,450000]}
df=pd.DataFrame(data)
df_filled=df.fillna(method='ffill')#replace the previous row's value(f-fill na forward fill)
print(df_filled)


import pandas as pd
import numpy as np
data={'name':['John','Alice','Bob','charlie','Dave','Eve','Bob','John'],
      'age':[25,23,22,np.nan,np.nan,26,22,np.nan],
      'gender':['M','M','F',np.nan,'F',np.nan,'F','F'],
      'salary':[500000,100000,700000,np.nan,10000,20000,700000,450000]}
df=pd.DataFrame(data)
df_filled=df.fillna(method='bfill')#replace the next row's value(b-fill na forward fill)
print(df_filled)
#if the last value is nan then it will not b replaced with any value (exceptional case)

import pandas as pd
import numpy as np
data={'name':['John','Alice','Bob','charlie','Dave','Eve','Bob','John'],
      'age':[25,23,22,np.nan,np.nan,26,22,np.nan],
      'gender':['M','M','F',np.nan,'F',np.nan,'F','F'],
      'salary':[500000,100000,700000,np.nan,10000,20000,700000,450000]}
df=pd.DataFrame(data)
df_mean=df.fillna(df.mean())
print(df_mean)
#missing value is replaced by average value

import pandas as pd
import numpy as np
data={'name':['John','Alice','Bob','charlie','Dave','Eve','Bob','John'],
      'age':[25,23,22,np.nan,np.nan,26,22,np.nan],
      'gender':['M','M','F',np.nan,'F',np.nan,'F','F'],
      'salary':[500000,100000,700000,np.nan,10000,20000,700000,450000]}
df=pd.DataFrame(data)
df_mean=df.fillna(df['age'].mean())
print(df_mean)


# Result
        the given data has been performed data cleaning and saved the cleaned data to a file successfully.
