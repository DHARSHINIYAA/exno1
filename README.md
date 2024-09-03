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
df=pd.read_csv("SAMPLEIDS.csv")
df
df.info()
```
![image](https://github.com/user-attachments/assets/5ac7146b-9622-4f81-b8fc-fd7693961732)

```
import pandas as pd
df=pd.read_csv("SAMPLEIDS.csv")
print(df)
df.describe()
```
![Screenshot 2024-08-28 084855](https://github.com/user-attachments/assets/1dc978c2-444f-4c61-a4f9-dc5040d13c34)
```
import pandas as pd
df=pd.read_csv("SAMPLEIDS.csv")
df
df.head(5)
```
![Screenshot 2024-08-28 085024](https://github.com/user-attachments/assets/fa5dbd7d-c33f-45d4-9c27-b57d8430e4f9)

```
import pandas as pd
df=pd.read_csv("SAMPLEIDS.csv")
df
df.tail(5)
```
![image](https://github.com/user-attachments/assets/5e035fc6-2362-45fd-8658-83cd117ef11d)

```
df.isnull().sum()#displays how many missing value
```
![image](https://github.com/user-attachments/assets/1c633f03-f506-4959-a227-d7da60f04023)
```
import pandas as pd
df=pd.read_csv("SAMPLEIDS.csv")
df
df.nunique#displays unique value counts
```
![image](https://github.com/user-attachments/assets/ed95da65-8985-489a-b4b6-7ac3bb7916fa)
```
import pandas as pd
df=pd.read_csv("SAMPLEIDS.csv")
df
df['GENDER'].value_counts()#counts the no.of male and female in particular column
```
![image](https://github.com/user-attachments/assets/3f55e163-2f67-4e9f-a260-b77106ce9909)
```
import pandas as pd
df=pd.read_csv("SAMPLEIDS.csv")
df
df.dropna(how='any').shape#rows and columns with missing value are deleted and then counted 
```
![image](https://github.com/user-attachments/assets/976d3543-586a-4ef5-a639-9207fe3f529c)

```
import pandas as pd
df=pd.read_csv("SAMPLEIDS.csv")
df
df.shape#all rows and columns are counted
```
![image](https://github.com/user-attachments/assets/f9742571-7a19-4188-8529-2a2bbc6c136b)
```

import pandas as pd
df=pd.read_csv("SAMPLEIDS.csv")
df
x=df.dropna(how='any')#missing values are deleted and rows with all value are printed
print(x)
```
![image](https://github.com/user-attachments/assets/66fa4ff7-04c0-470f-8bbf-06f2bb9239d8)
```
import pandas as pd
df=pd.read_csv("SAMPLEIDS.csv")
df
x1=df.dropna(how='all')#if elements of rows is completely NULL then it is deleted
print(x1)
```
![image](https://github.com/user-attachments/assets/f6de1296-e511-4063-a68a-2226e44ee40b)
```
import pandas as pd
df=pd.read_csv("SAMPLEIDS.csv")
df
tot=df.dropna(subset=['TOTAL'],how='any')#missing values are deleted from total then it is printed 
print(tot)
df.shape
```
![image](https://github.com/user-attachments/assets/7dea361e-c349-459d-86a1-a3c21bd4c03b)

```
import pandas as pd
df=pd.read_csv("SAMPLEIDS.csv")
df
tot=df.dropna(subset=['M1','M2','M3','M4'],how='any')#missing values are deleted from total then it is printed 
print(tot)
df.shape
```
![image](https://github.com/user-attachments/assets/79cbe6bd-1a8f-4ff9-a689-0d9fa2c9a683)
```
import pandas as pd
df=pd.read_csv("SAMPLEIDS.csv")
df
s=df.fillna(0)#filling the missing value with 0
print(s)
```
![image](https://github.com/user-attachments/assets/4604a866-be04-4eb4-aa93-ac0e6d80e69f)
```
import pandas as pd
df=pd.read_csv("SAMPLEIDS.csv")
df
df.isna().sum()
```
![image](https://github.com/user-attachments/assets/1ad153fb-0f7c-41d2-a5a1-4fc92fb77691)
```
import pandas as pd
df=pd.read_csv("SAMPLEIDS.csv")
df
df['M1']
```
![image](https://github.com/user-attachments/assets/a20ea9cc-8641-4a64-a6bb-851f440ff6c2)
```
import pandas as pd
df=pd.read_csv("SAMPLEIDS.csv")
df
df.isnull()#missing value -t,not missing value-f
```
![image](https://github.com/user-attachments/assets/0a9dd7f5-ae8b-43ba-a626-2cf6990e6e72)

```
import pandas as pd
df=pd.read_csv("SAMPLEIDS.csv")
df
~df.isnull()
```
![image](https://github.com/user-attachments/assets/ce43f6be-cf47-49f3-9059-53969751fe53)
```
import pandas as pd
df=pd.read_csv("SAMPLEIDS.csv")
df
df.notnull()#(~ and not null are same)
```
![image](https://github.com/user-attachments/assets/4c1f02c0-a6cd-41cb-b90d-29d6b26f3653)
```
import pandas as pd
df=pd.read_csv("SAMPLEIDS.csv")
df
x1=df.dropna(axis=0,inplace=True)
print(x1)
```
![image](https://github.com/user-attachments/assets/f3d199ee-dd36-4775-b4ad-235c06dc021e)
```
import pandas as pd
df=pd.read_csv("SAMPLEIDS.csv")
df
x1=df.dropna(axis=0,inplace=False)
print(x1)
```
![Screenshot 2024-08-28 090839](https://github.com/user-attachments/assets/46e336c8-0117-4175-8d15-e57c748d60eb)
```
import pandas as pd
df=pd.read_csv("SAMPLEIDS.csv")
df
x1=df.dropna(axis=0)
print(x1)
```
![image](https://github.com/user-attachments/assets/1f2378db-4370-4414-94bf-c3e4c1c91ac5)
```
import pandas as pd
df=pd.read_csv("SAMPLEIDS.csv")
df
df.duplicated()
```
![image](https://github.com/user-attachments/assets/22735363-c31b-4899-a6bf-14757f0de9a0)
```
import pandas as pd
df=pd.read_csv("SAMPLEIDS.csv")
df
#x1=df.drop_duplicates(inplace=True) #output-none
x1=df.drop_duplicates(inplace= False)
print(x1)
```
![image](https://github.com/user-attachments/assets/f463efa2-af50-4cbe-8046-e75b8f864792)

```
import seaborn as sns
sns.heatmap(df.isnull(),yticklabels=False,annot=True)
```
![image](https://github.com/user-attachments/assets/71c279a9-d413-4e25-9b6c-8c6edaffd3d0)

```
import pandas as pd
df=pd.read_csv("SAMPLEIDS.csv")
df
print(df.loc[0:3])
```
![image](https://github.com/user-attachments/assets/284a578e-ea82-400c-80ba-3c5c26e19052)
```
import pandas as pd
df=pd.read_csv("SAMPLEIDS.csv")
df.dropna(inplace=True)
sns.heatmap(df.isnull(),yticklabels=False,annot=True)
```
![image](https://github.com/user-attachments/assets/2e79a6db-53aa-487f-97f3-7eafd4696138)
```
import pandas as pd
import seaborn as sns
import numpy as np
age={1,3,28,27,25,92,30,39,40,50,26,24,29,94}
af=pd.DataFrame(age)
print(af)
```
![image](https://github.com/user-attachments/assets/3bf79f96-c203-410a-8d6e-09e27c428620)

```
sns.boxplot(data=af)#outlier are 1,3 below the blue box and 92,94 are outlier above the blue box
#outlier - to high or to low from the given range 
```
![image](https://github.com/user-attachments/assets/92b58984-53e1-4f89-9c18-a67a864a9bc1)
```
sns.boxenplot(data=af)#to understand whether the outlier exist or not
```
![image](https://github.com/user-attachments/assets/5e6162c4-a434-42c9-b3c9-e1ca89ea1149)
```
sns.scatterplot(data=af)
```
![image](https://github.com/user-attachments/assets/17c9c997-4cdf-4960-9a68-95f2d269adf6)
```
#iqr-inter quantile reduction
q1=af.quantile(0.25)
q2=af.quantile(0.5)
q3=af.quantile(0.75)
iqr=q3-q1
print(iqr)
```
![image](https://github.com/user-attachments/assets/a38d1831-eee9-489d-a60a-4abd33fefa57)
```
#either this or previous one
q1=np.percentile(af,25)
q3=np.percentile(af,75)
iqr=q3-q1
print(iqr)
```
![image](https://github.com/user-attachments/assets/778fd511-73c6-4e2a-b9ec-e138b328b3a9)
```
lower_bound=q1-1.5*iqr
print(lower_bound)
```
![image](https://github.com/user-attachments/assets/15f8e3e4-b9a2-4c7e-98c4-62b86ad9ff65)
```
upper_bound=q3+1.5*iqr
print(upper_bound)
```
![image](https://github.com/user-attachments/assets/bddef5a7-3f49-4c2e-b609-9e13a8832125)
```
outliers=[x for x in age if x <lower_bound or x > upper_bound]
print(outliers)
```
![image](https://github.com/user-attachments/assets/b1f84937-60ae-4454-9eb5-4b0969569a34)
```
print(iqr)
print("Q1:",q1)
print("Q3:",q3)
print("lower_bound:",lower_bound)
print("upper_bound",upper_bound)
print("outliers",outliers)
```
![image](https://github.com/user-attachments/assets/869d00e9-dff8-4eb0-8efa-c8e3a599f454)
```
af=af[(af>=lower_bound)&(af>=lower_bound)]
print(af)
```
![image](https://github.com/user-attachments/assets/31d670c3-5c17-4acf-8657-e6f5c6e9f096)

```
id=pd.read_csv("iris.csv")
id
```

![image](https://github.com/user-attachments/assets/db0f58fb-b3af-4974-9e0d-1abbdf8302eb)

```

sns.boxplot(x='sepal_width',data=id)
```




# Result:
Thus we have cleaned the data and removed the outliers by detection using IQR and Z-score method.
