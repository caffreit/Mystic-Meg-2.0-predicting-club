
# Predicting if someone is a member of a club or not.


```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
%matplotlib inline
import math
```


```python
path_to_file = 'C:\Users\Administrator\Desktop\Python Scripts\examplepark.csv'
data = pd.read_csv(path_to_file)
data['Time'] = ((pd.to_numeric(data['Time'].str.slice(0,2)))*60)+(pd.to_numeric\
(data['Time'].str.slice(3,5)))+((pd.to_numeric(data['Time'].str.slice(6,8)))/60)
data['Date'] = pd.to_datetime(data['Date'],errors='coerce', format='%d-%m-%Y')
data['Age_Cat'] = pd.to_numeric(data['Age_Cat'].str.slice(2,4),errors='coerce', downcast='integer')
data['Age_Grade'] = pd.to_numeric(data['Age_Grade'].str.slice(0,5),errors='coerce')
data['Club_Coded'] = data['Club'].isnull()
```

### Recoding and Shuffling the data


```python
def converter(Club):
    try:
        Club=float(Club)
        return 0
    except:
        return 1
```


```python
data['Club_Coded'] = data['Club'].apply(converter)
from sklearn.utils import shuffle
data = shuffle(data)
data.head(10)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Pos</th>
      <th>Name</th>
      <th>Time</th>
      <th>Age_Cat</th>
      <th>Age_Grade</th>
      <th>Gender</th>
      <th>Gen_Pos</th>
      <th>Club</th>
      <th>Note</th>
      <th>Total_Runs</th>
      <th>Run_No.</th>
      <th>Club_Coded</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>40725</th>
      <td>2015-03-07</td>
      <td>244</td>
      <td>Rita GARDINER</td>
      <td>29.400000</td>
      <td>45.0</td>
      <td>54.54</td>
      <td>F</td>
      <td>68.0</td>
      <td>Sloggers to Joggers</td>
      <td>PB stays at 00.25.40</td>
      <td>31.0</td>
      <td>122</td>
      <td>1</td>
    </tr>
    <tr>
      <th>19324</th>
      <td>2014-01-11</td>
      <td>382</td>
      <td>Unknown</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>62</td>
      <td>0</td>
    </tr>
    <tr>
      <th>444</th>
      <td>2012-11-24</td>
      <td>70</td>
      <td>Brid BEAUSANG</td>
      <td>23.200000</td>
      <td>35.0</td>
      <td>64.66</td>
      <td>F</td>
      <td>6.0</td>
      <td>NaN</td>
      <td>First Timer!</td>
      <td>66.0</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>38050</th>
      <td>2015-01-17</td>
      <td>373</td>
      <td>Madeleine MANGAN</td>
      <td>33.050000</td>
      <td>55.0</td>
      <td>54.77</td>
      <td>F</td>
      <td>139.0</td>
      <td>NaN</td>
      <td>PB stays at 00.26.12</td>
      <td>16.0</td>
      <td>115</td>
      <td>0</td>
    </tr>
    <tr>
      <th>80074</th>
      <td>2017-09-16</td>
      <td>127</td>
      <td>Aoife SLATER</td>
      <td>27.550000</td>
      <td>30.0</td>
      <td>54.08</td>
      <td>F</td>
      <td>33.0</td>
      <td>Forget The Gym</td>
      <td>PB stays at 00.24.32</td>
      <td>105.0</td>
      <td>254</td>
      <td>1</td>
    </tr>
    <tr>
      <th>27821</th>
      <td>2014-06-21</td>
      <td>122</td>
      <td>Declan O'BRIEN</td>
      <td>25.150000</td>
      <td>45.0</td>
      <td>58.18</td>
      <td>M</td>
      <td>103.0</td>
      <td>NaN</td>
      <td>New PB!</td>
      <td>26.0</td>
      <td>85</td>
      <td>0</td>
    </tr>
    <tr>
      <th>76431</th>
      <td>2017-06-10</td>
      <td>109</td>
      <td>Unknown</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>240</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7583</th>
      <td>2013-05-04</td>
      <td>456</td>
      <td>Zoe CRUISE</td>
      <td>36.616667</td>
      <td>25.0</td>
      <td>40.42</td>
      <td>F</td>
      <td>208.0</td>
      <td>NaN</td>
      <td>PB stays at 00.28.28</td>
      <td>17.0</td>
      <td>26</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19759</th>
      <td>2014-01-25</td>
      <td>114</td>
      <td>Mai BARRETT</td>
      <td>25.633333</td>
      <td>35.0</td>
      <td>59.69</td>
      <td>F</td>
      <td>16.0</td>
      <td>Portmarnock Athletic Club</td>
      <td>PB stays at 00.23.50</td>
      <td>98.0</td>
      <td>64</td>
      <td>1</td>
    </tr>
    <tr>
      <th>54012</th>
      <td>2015-12-26</td>
      <td>115</td>
      <td>Unknown</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>164</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df = data[:5000]
df = df.drop('Club',1)
df = df.dropna()
df = df.drop('Gender',1)
df = df.drop('Note',1)
df = df.drop('Name',1)
df = df.drop('Date',1)
df = df.drop('Gen_Pos',1)
df = df.drop('Age_Grade',1)

cols_to_norm = ['Pos', 'Time', 'Age_Cat', 'Total_Runs', 'Run_No.']
df[cols_to_norm] = df[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pos</th>
      <th>Time</th>
      <th>Age_Cat</th>
      <th>Total_Runs</th>
      <th>Run_No.</th>
      <th>Club_Coded</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>40725</th>
      <td>0.426316</td>
      <td>0.287195</td>
      <td>0.500000</td>
      <td>0.084507</td>
      <td>0.441606</td>
      <td>1</td>
    </tr>
    <tr>
      <th>444</th>
      <td>0.121053</td>
      <td>0.162822</td>
      <td>0.357143</td>
      <td>0.183099</td>
      <td>0.007299</td>
      <td>0</td>
    </tr>
    <tr>
      <th>38050</th>
      <td>0.652632</td>
      <td>0.360415</td>
      <td>0.642857</td>
      <td>0.042254</td>
      <td>0.416058</td>
      <td>0</td>
    </tr>
    <tr>
      <th>80074</th>
      <td>0.221053</td>
      <td>0.250084</td>
      <td>0.285714</td>
      <td>0.292958</td>
      <td>0.923358</td>
      <td>1</td>
    </tr>
    <tr>
      <th>27821</th>
      <td>0.212281</td>
      <td>0.201939</td>
      <td>0.500000</td>
      <td>0.070423</td>
      <td>0.306569</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
features = df.drop('Club_Coded', axis=1)

X = features
y = df['Club_Coded']
```

# Random Forest
### Import libraries and splitting data into training and test data.


```python
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.grid_search import GridSearchCV

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

    C:\ProgramData\Anaconda2\lib\site-packages\sklearn\cross_validation.py:44: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
      "This module will be removed in 0.20.", DeprecationWarning)
    C:\ProgramData\Anaconda2\lib\site-packages\sklearn\grid_search.py:43: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. This module will be removed in 0.20.
      DeprecationWarning)
    


```python
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train,y_train)
rfc_preds = rfc.predict(X_test)

print(confusion_matrix(y_test,rfc_preds))
print('\n')
print(classification_report(y_test,rfc_preds))
```

    [[824  73]
     [218 142]]
    
    
                 precision    recall  f1-score   support
    
              0       0.79      0.92      0.85       897
              1       0.66      0.39      0.49       360
    
    avg / total       0.75      0.77      0.75      1257
    
    

## Optimising Hyperparameter


```python
from sklearn.model_selection import RandomizedSearchCV
from pprint import pprint
```


```python
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 5)]
max_features = ['auto']
max_depth = [int(x) for x in np.linspace(5, 25, num = 5)]
max_depth.append(None)
min_samples_split = [5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
```


```python
pprint(random_grid)
```

    {'bootstrap': [True, False],
     'max_depth': [5, 10, 15, 20, 25, None],
     'max_features': ['auto'],
     'min_samples_leaf': [1, 2, 4],
     'min_samples_split': [5, 10],
     'n_estimators': [100, 575, 1050, 1525, 2000]}
    


```python
rf_random = RandomizedSearchCV(estimator = rfc, param_distributions = random_grid, n_iter = 50,
                               cv = 3, verbose=0, random_state=42, n_jobs = 2)
rf_random.fit(X_train,y_train)
```




    RandomizedSearchCV(cv=3, error_score='raise',
              estimator=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_impurity_split=1e-07, min_samples_leaf=1,
                min_samples_split=2, min_weight_fraction_leaf=0.0,
                n_estimators=200, n_jobs=1, oob_score=False, random_state=None,
                verbose=0, warm_start=False),
              fit_params={}, iid=True, n_iter=50, n_jobs=2,
              param_distributions={'bootstrap': [True, False], 'min_samples_leaf': [1, 2, 4], 'n_estimators': [100, 575, 1050, 1525, 2000], 'min_samples_split': [5, 10], 'max_features': ['auto'], 'max_depth': [5, 10, 15, 20, 25, None]},
              pre_dispatch='2*n_jobs', random_state=42, refit=True,
              return_train_score=True, scoring=None, verbose=0)




```python
rf_random.best_params_
```




    {'bootstrap': False,
     'max_depth': 25,
     'max_features': 'auto',
     'min_samples_leaf': 1,
     'min_samples_split': 5,
     'n_estimators': 1050}




```python
grid_predictions = rf_random.predict(X_test)
print(confusion_matrix(y_test,grid_predictions))
print('\n')
print(classification_report(y_test,grid_predictions))
```

    [[869  75]
     [171 258]]
    
    
                 precision    recall  f1-score   support
    
              0       0.84      0.92      0.88       944
              1       0.77      0.60      0.68       429
    
    avg / total       0.82      0.82      0.81      1373
    
    

### Running the model on a much larger dataset.


```python
df1 = data[:75000]
df1 = df1.drop('Club',1)
df1 = df1.dropna()
df1 = df1.drop('Gender',1)
df1 = df1.drop('Note',1)
df1 = df1.drop('Name',1)
df1 = df1.drop('Date',1)
df1 = df1.drop('Gen_Pos',1)
df1 = df1.drop('Age_Grade',1)
df1.tail()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pos</th>
      <th>Time</th>
      <th>Age_Cat</th>
      <th>Total_Runs</th>
      <th>Run_No.</th>
      <th>Club_Coded</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3588</th>
      <td>173</td>
      <td>27.383333</td>
      <td>45.0</td>
      <td>69.0</td>
      <td>16</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18034</th>
      <td>116</td>
      <td>26.033333</td>
      <td>40.0</td>
      <td>97.0</td>
      <td>57</td>
      <td>0</td>
    </tr>
    <tr>
      <th>35344</th>
      <td>109</td>
      <td>24.583333</td>
      <td>15.0</td>
      <td>80.0</td>
      <td>107</td>
      <td>0</td>
    </tr>
    <tr>
      <th>30941</th>
      <td>94</td>
      <td>23.766667</td>
      <td>40.0</td>
      <td>105.0</td>
      <td>95</td>
      <td>1</td>
    </tr>
    <tr>
      <th>46266</th>
      <td>240</td>
      <td>30.916667</td>
      <td>11.0</td>
      <td>19.0</td>
      <td>136</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
features = df1.drop('Club_Coded', axis=1)

X = features
y = df1['Club_Coded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```


```python
rfc = RandomForestClassifier(n_estimators=1000, max_depth=25, max_features='auto', 
                             min_samples_leaf=1, min_samples_split=5) ### ad in the good parmas
rfc.fit(X_train,y_train)
rfc_preds = rfc.predict(X_test)

print(confusion_matrix(y_test,rfc_preds))
print('\n')
print(classification_report(y_test,rfc_preds))
```

    [[13085   502]
     [ 2212  3141]]
    
    
                 precision    recall  f1-score   support
    
              0       0.86      0.96      0.91     13587
              1       0.86      0.59      0.70      5353
    
    avg / total       0.86      0.86      0.85     18940
    
    

### Testing against the holdout data


```python
holdout_data = data[75001:]
holdout_data = holdout_data.drop('Club',1)
holdout_data = holdout_data.dropna()
holdout_data = holdout_data.drop('Gender',1)
holdout_data = holdout_data.drop('Note',1)
holdout_data = holdout_data.drop('Name',1)
holdout_data = holdout_data.drop('Date',1)
holdout_data = holdout_data.drop('Gen_Pos',1)
holdout_data = holdout_data.drop('Age_Grade',1)
holdout_data.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pos</th>
      <th>Time</th>
      <th>Age_Cat</th>
      <th>Total_Runs</th>
      <th>Run_No.</th>
      <th>Club_Coded</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>46613</th>
      <td>288</td>
      <td>33.766667</td>
      <td>18.0</td>
      <td>77.0</td>
      <td>137</td>
      <td>0</td>
    </tr>
    <tr>
      <th>55595</th>
      <td>179</td>
      <td>28.233333</td>
      <td>10.0</td>
      <td>17.0</td>
      <td>169</td>
      <td>0</td>
    </tr>
    <tr>
      <th>62006</th>
      <td>196</td>
      <td>29.116667</td>
      <td>30.0</td>
      <td>8.0</td>
      <td>189</td>
      <td>0</td>
    </tr>
    <tr>
      <th>85967</th>
      <td>311</td>
      <td>40.266667</td>
      <td>40.0</td>
      <td>15.0</td>
      <td>274</td>
      <td>0</td>
    </tr>
    <tr>
      <th>35387</th>
      <td>152</td>
      <td>26.416667</td>
      <td>50.0</td>
      <td>78.0</td>
      <td>107</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
holdout_data_feat = holdout_data.drop('Club_Coded', axis=1)

y_holdout = holdout_data['Club_Coded']
X_holdout = holdout_data_feat
```


```python
rfc_preds = rfc.predict(X_holdout)

print(confusion_matrix(y_holdout,rfc_preds))
print('\n')
print(classification_report(y_holdout,rfc_preds))
```

    [[6521  249]
     [1166 1500]]
    
    
                 precision    recall  f1-score   support
    
              0       0.85      0.96      0.90      6770
              1       0.86      0.56      0.68      2666
    
    avg / total       0.85      0.85      0.84      9436
    
    

### Feature Importance


```python
feature_list = ['Pos', 'Time', 'Age_Cat', 'Total_Runs', 'Run_No.']
importances = list(rfc.feature_importances_)
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
print feature_importances
```

    [('Total_Runs', 0.32), ('Time', 0.22), ('Run_No.', 0.18), ('Pos', 0.16), ('Age_Cat', 0.12)]
    


```python

```
