from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from six.moves import urllib

import tensorflow.compat.v2.feature_column as fc
import tensorflow as tf

dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') # training data
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv') # testing data
#print(dftrain.head())  #head shows first five entries
y_train = dftrain.pop('survived')   #removes survived column from the dftrain and stores it in y_train as we need to separate the training data and the one on which we perfom operation
y_eval = dfeval.pop('survived')

#print(dftrain.head())
#print(y_train.head())

#print(dftrain["age"])
#print(dftrain.loc[0])

#print(dftrain.describe()) #describes the dataset
#print(dftrain.shape)  #shape of the dataframe

his = dftrain.age.hist(bins=20)      #histogram for ages
#print(his)

sx = dftrain.sex.value_counts().plot(kind='barh')   #bar graph for sex count
#print(sx)

CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',  #uppercase signifies constant
                       'embark_town', 'alone']

NUMERIC_COLUMNS = ['age', 'fare']

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
  vocabulary = dftrain[feature_name].unique()  # gets a list of all unique values from given feature column
  feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
  feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

#print(feature_columns)

def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
  def input_function():  # inner function, this will be returned
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))  # create tf.data.Dataset object with data and its label
    if shuffle:
      ds = ds.shuffle(1000)  # randomize order of data
    ds = ds.batch(batch_size).repeat(num_epochs)  # split dataset into batches of 32 and repeat process for number of epochs
    return ds  # return a batch of the dataset
  return input_function  # return a function object for use

train_input_fn = make_input_fn(dftrain, y_train)  # here we will call the input_function that was returned to us to get a dataset object we can feed to the model
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)
# We create a linear estimtor by passing the feature columns we created earlier

linear_est.train(train_input_fn)  # train
result = linear_est.evaluate(eval_input_fn)  # get model metrics/stats by testing on tetsing data

#clear_output()  # clears consoke output
#print(result['accuracy'])  # the result variable is simply a dict of stats about our model
#print(result)

rs = list(linear_est.predict(eval_input_fn))
print(dfeval.loc[5])
print(y_eval.loc[5])
print(rs[5]['probabilities'][0])