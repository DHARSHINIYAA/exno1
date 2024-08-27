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
```
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

import pandas as pd
df=pd.read_csv('SAMPLEIDS.csv')
df

import pandas as pd
df=pd.read_csv('SAMPLEIDS.csv')
df
df.info()


import pandas as pd
df=pd.read_csv('SAMPLEIDS.csv')
print(df)
print(df.describe())


import pandas as pd
df=pd.read_csv('SAMPLEIDS.csv')
df
df.head(5)

import pandas as pd
df=pd.read_csv('SAMPLEIDS.csv')
df
df.tail(5)

import pandas as pd
df=pd.read_csv('SAMPLEIDS.csv')
df
df.isnull().sum()#display how many missing value

import pandas as pd
df=pd.read_csv('SAMPLEIDS.csv')
df
df.nunique

import pandas as pd
df=pd.read_csv('SAMPLEIDS.csv')
df
df['GENDER'].value_counts()

import pandas as pd
df=pd.read_csv('SAMPLEIDS.csv')
df
df.dropna(how='any').shape

import pandas as pd
df=pd.read_csv('SAMPLEIDS.csv')
df
df.shape#total rows nd column

import pandas as pd
df=pd.read_csv('SAMPLEIDS.csv')
df
x=df.dropna(how='any')#if any 1 value is missing del the whole row
print(x)

import pandas as pd
df=pd.read_csv('SAMPLEIDS.csv')
df
x2=df.dropna(how='all').shape#if all the values r missing then del the row

import pandas as pd
df=pd.read_csv('SAMPLEIDS.csv')
df
tot=df.dropna(subset=['TOTAL'],how='any')
print(tot)

import pandas as pd
df=pd.read_csv('SAMPLEIDS.csv')
df
tot=df.dropna(subset=['M1','M2','M3','M4'],how='any')
print(tot)

import pandas as pd
df=pd.read_csv('SAMPLEIDS.csv')
df
s=df.fillna(0)
s

import pandas as pd
df=pd.read_csv('SAMPLEIDS.csv')
df
df.isna().sum()

import pandas as pd
df=pd.read_csv('SAMPLEIDS.csv')
df
df['M1']

import pandas as pd
df=pd.read_csv('SAMPLEIDS.csv')
df
df.isnull()#to know missing value

import pandas as pd
df=pd.read_csv('SAMPLEIDS.csv')
df
df.notnull()#to know non missing value

import pandas as pd
df=pd.read_csv('SAMPLEIDS.csv')
df
x1=df.dropna(axis=0)
print(x1)

import pandas as pd
df=pd.read_csv('SAMPLEIDS.csv')
df
df.duplicated()

import pandas as pd
df=pd.read_csv('SAMPLEIDS.csv')
df
m=df.drop_duplicates(inplace=False)#to fine duplicates nd drop them, if true it executes, false not executes
m

import pandas as pd
df=pd.read_csv('SAMPLEIDS.csv')
import seaborn as sns
sns.heatmap(df.isnull(),yticklabels=False,annot=True)   

import pandas as pd
df=pd.read_csv('SAMPLEIDS.csv')
df.dropna(inplace=True)
sns.heatmap(df.isnull(),yticklabels=False,annot=True)

import pandas as pd
df=pd.read_csv('SAMPLEIDS.csv')
print(df.loc[0:3])

import pandas as pd
df=pd.read_csv('SAMPLEIDS.csv')
df.dtypes

import pandas as pd
df=pd.read_csv('SAMPLEIDS.csv')
df.filter(regex='a',axis=1)

#import pandas as pd
#de=pd.read_excel('STUDENT.xlsx',sheet_name='DET')
#to add excel files

import pandas as pd
import seaborn as sns
import numpy as np
age=[1,2,3,4,5,6,7,8,9,10,25,20,36,40,35,33]
af=pd.DataFrame(age)
af

import pandas as pd
import seaborn as sns
import numpy as np
age=[1,3,28,27,25,92,30,39,40,50,26,24,29,94]
af=pd.DataFrame(age)
sns.boxplot(data=af)

import pandas as pd
import seaborn as sns
import numpy as np
age=[1,3,28,27,25,92,30,39,40,50,26,24,29,94]
af=pd.DataFrame(age)
sns.boxenplot(data=af)

import pandas as pd
import seaborn as sns
import numpy as np
age=[1,3,28,27,25,92,30,39,40,50,26,24,29,94]
af=pd.DataFrame(age)
sns.scatterplot(data=af)

q1=af.quantile(0.25)
q2=af.quantile(0.5)
q3=af.quantile(0.75)
iqr=q3-q1
print(iqr)

Q1=np.percentile(af,25)
Q3=np.percentile(af,75)
IQR=Q3-Q1
print(IQR)

lower_bound=Q1-1.5*IQR
print(lower_bound)

upper_bound=Q3+1.5*IQR
print(upper_bound)

outliers=[x for x in age if x< lower_bound or x > upper_bound]
print(outliers)

print("Q1:",Q1)
print("Q3:",Q3)
print("IQR:",IQR)
print("lower bound:",lower_bound)
print("upper bound:",upper_bound)
print("outliers:",outliers)

af=af[((af>=lower_bound)&(af<=upper_bound))]
print(af)
#outliers r replaced by nan

af.dropna()

sns.boxplot(data=af)


```

# Result
        the given data has been performed data cleaning and saved the cleaned data to a file successfully.
