---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.2
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

```{code-cell} ipython3
:id: aAAH9q7UP_yy

import pandas as pd
import matplotlib as plt
import seaborn as sns
import random
```

```{code-cell} ipython3
:id: EA4dJBTlQGIh

# Let us start with a simple dataset
x = [i for i in range(0,22, 2)]
y = [4+0.8*i+random.random() for i in range(22, 0, -2) ]

# Dataframe
df = pd.DataFrame(
    {'x': x,
     'y': y}
)
```

```{code-cell} ipython3
:id: qcwiA5EFVn8X

df.head()
```

```{code-cell} ipython3
:id: qkIjCotFUJns

sns.scatterplot(x='x', y = 'y', data=df)
```

+++ {"id": "ykIA3ds4zvn2"}

Let us look at the correlation between x and y. Pandas has a corr method that allows to find the correlation between two columns. For demo puposes, we created the dataset to be highly negatively correlated. Therefore, we should expect a number close to -1, as we are using Pearson's correlation.  

```{code-cell} ipython3
:id: LtkIzPnSyjfc

df.x.corr(df.y, method='pearson')
```

```{code-cell} ipython3
:id: 3rVeQz-tVY81

# The mean of x and y
x_mean = df.x.mean()
y_mean = df.y.mean()


df['xycov'] = (df['x'] - x_mean) * (df['y'] - y_mean)
df['xvar'] = (df['x'] - x_mean)**2

# Calculate the slope and intercept
m = df['xycov'].sum() / df['xvar'].sum()
c = y_mean - (m * x_mean)
print('c = ', c)
print ('m = ', m)
print ('line: ', 'y = '+str(round(c, 3))+'x + '+str(round(m,3)))
```

```{code-cell} ipython3
:id: D8SS8BXnxhEa

# Now we can use the line to predict y
df['y_pred'] = m*df['x']+c
```

```{code-cell} ipython3
:id: NTXU6lUIVYnK

df
```

```{code-cell} ipython3
:id: MW-aeHorzVF8

# Lets plot the line
sns.lineplot(x='x', y = 'y_pred', data=df, color='red')
sns.scatterplot(x='x', y = 'y', data=df)
```

```{code-cell} ipython3
:id: x5smhDfBkQUu

# Let us plot both y and y_pred to have a visual sense of how we did
df.plot(kind='bar', x='x', y=['y', 'y_pred'], figsize=(20, 5))
```

```{code-cell} ipython3
:id: I7Ymt_qGrKrg

# Calculate the coefficient of determination or R squared
# The coefficient of determination is the proportion of the variance in the dependent variable that is predictable from the independent variable.
# It ranges from 0 to 1.

1 - ((df['y'] - df['y_pred'])).var()/df['y'].var()
```

+++ {"id": "GScGmt92LJPK"}

Here the predictability is really good.

```{code-cell} ipython3
:id: HahwZ0UA2Gex

# If we wanted to predict what y would be when a new point x=11 is given, we could
m*11+c
```

+++ {"id": "FSZqf8pc_FsH"}

Lets try out a real example

```{code-cell} ipython3
:id: 9JvxKsb1Qcjr

from google.colab import drive
drive.mount('/content/drive')
```

```{code-cell} ipython3
:id: pBGDYnDGQdJX

path = ""
```

```{code-cell} ipython3
:id: zDwQfQinQwBM

df = pd.read_csv(path)
```

```{code-cell} ipython3
:id: dFLZbvtmQ80W

df.head()
```

```{code-cell} ipython3
:id: SI0OuyJyfvcc

df.tail()
```

+++ {"id": "4EHKVMrt_VTU"}

Let us remove Achham and Udayapur for now. We will use the model we created to predict the proverty rate of Accham

```{code-cell} ipython3
:id: AGv4XX-V8G7q

df = df[1:-1]
```

```{code-cell} ipython3
:id: JSbe0d7K8Vi3

df.tail()
```

```{code-cell} ipython3
:id: 4FvorrKoQ45s

sns.scatterplot(x='literacy rate', y = 'poverty rate', data=df)
```

```{code-cell} ipython3
:id: jjtIBs5tzFDC

df['literacy rate'].corr(df['poverty rate'], method='pearson')
```

```{code-cell} ipython3
:id: Y4OYjrgURCsq

x_mean = df['literacy rate'].mean()
y_mean = df['poverty rate'].mean()


df['xycov'] = (df['literacy rate'] - x_mean) * (df['poverty rate'] - y_mean)
df['xvar'] = (df['literacy rate'] - x_mean)**2

# Calculate the slope and intercept
m = df['xycov'].sum() / df['xvar'].sum()
c = y_mean - (m * x_mean)
print('c = ', c)
print ('m = ', m)
print ('line: ', 'y = '+str(round(c, 3))+'x + '+str(round(m,3)))
```

```{code-cell} ipython3
:id: HeulYywk1XQ6

df['y_pred'] = m*df['literacy rate']+c
```

```{code-cell} ipython3
:id: 4AygGy4qRUqk

sns.lineplot(x='literacy rate', y = 'y_pred', data=df, color='red')
sns.scatterplot(x='literacy rate', y = 'poverty rate', data=df)
```

+++ {"id": "S9Mzg2kv1TYM"}

For Achham, the predicted poverty rate is

```{code-cell} ipython3
:id: yL4VP-_d2JPh

round(m*0.476151+c , 3)
```

+++ {"id": "CGzyptQBhnB2"}

But, we know that the actual poverty rate was 0.472

+++ {"id": "zFnuMP3xgE9x"}

For Udaypur, the predicted poverty rate is

```{code-cell} ipython3
:id: 0wvwnFY2AsSf

round (m*0.614868+c, 3)
```

+++ {"id": "OI0rbukPhEkc"}

But, we know that the actual poverty rate was 0.259

```{code-cell} ipython3
:id: WIUGNQTyjQKc

# coefficient of determination
(1 - ((df['poverty rate'] - df['y_pred'])).var()/df['poverty rate'].var())
```

+++ {"id": "v6W5oAtVwbxe"}

The accuracy or score of Linear Regression is tied to how well the data is correlated with each other. The coefficient of dertermination (R^2) for Linear Regression is the square of correlation.   

```{code-cell} ipython3
:id: ROc66aadQuSH


```
