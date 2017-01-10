from tensorflow.contrib import layers
import pandas as pd
import tensorflow as tf

dict = {'id':range(1,6), 'pclass':[1,2,1,2,1],
        'gender':['M','F','F','M','M'],
        'fare':[10.5,22.3,11.6,22.4,31.5]}
df = pd.DataFrame.from_dict(dict)
df.shape
df.info()

# Data processing to convert pandas-like input to tensorflow-like input

# continuous types - real valued column
id = layers.real_valued_column('id')
type(id)
id.key

fare = layers.real_valued_column('fare')
type(fare)
fare.key

# comprehension for creating all real valued columns to do above work efficiently
cont_features = ['id','fare']
cont_feature_cols = [layers.real_valued_column(k) for k in cont_features]
# input features are in a list format
                     
# bucketized columns
# converting a continuous attribute to categorical/bucketized features
fare_buckets = layers.bucketized_column(fare,boundaries=[15,30])
type(fare_buckets)
fare_buckets.key

# converting continuous valued feature data to tensors
df['id'] # equivalent to type(df.id)
df[['id']]
type(df['id']) # Series
type(df[['id']]) # DataFrame
df[['id']].size
type(df[['id']].values)
ct = tf.constant(df[['id']].values)
type(ct)

# no [[]] here because we are using an element of a list (k)
# which itself is like list[]
cont_features_tensor = {k: tf.constant(df[k].values) for k in cont_features}
# data is in dictionary format                        

