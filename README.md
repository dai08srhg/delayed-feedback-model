# delayed-feedback-model

## Overview
PyTorch implementation of the paper.  
[Modeling Delayed Feedback in Display Advertising, Olivier Chapelle, KDD2014](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.650.6087&rep=rep1&type=pdf)

## Dataset
columns
- `feature1` ... `feature_n`: Categorical feautre column. (Assuming all variables are categorical.)
- `elapsed_day`: Days elapsed since click.
- `cv_delay_day`: Days delayed from click to conversion. (Only observable if conversions are observed)
- `supervised`: Conversion label. (If conversions are observed 1)

sample dataset
```
   feature1  feature2  feature3  elapsed_day  cv_delay_day  supervised
0         1         1         1           10           3.0           1
1         3         3         3            3           NaN           0
2         5         5         5           30           NaN           0
3         7         7         7            2           1.0           1
4         2         2         2            6           NaN           0
5         5         5         5            1           NaN           0
6         1         1         1           11           8.0           1
7         3         3         3           32           NaN           0
```

## remarks
In the paper, [feature hashing](https://arxiv.org/abs/0902.2206) is used for vectorization of categorical variables.
> All the features are mapped into a sparse binary feature vector of dimension 2^24 via the hashing trick [17].


In this implementation, embedding layer is used instead of feature hashing for vectorization.