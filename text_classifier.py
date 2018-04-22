import tensorflow as tf
import tensorflow_hub as hub
# import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
# import seaborn as sns

import csv

score = "sOPN"

# Load all files from a directory in a DataFrame.
def load_facebook_data():
    data = {}
    data["status"] = []
    data[score] = []
    # data["neu"] = []
    # data["agr"] = []
    # data["con"] = []
    # data["opn"] = []
    with open('facebook_clean.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data["status"].append(row["STATUS"])
            data[score].append( 1 if float(row[score]) > 2.5 else 0)
            # data["neu"].append(row["sNEU"])
            # data["agr"].append(row["sAGR"])
            # data["con"].append(row["sCON"])
            # data["opn"].append(row["sOPN"])\
    train = {}
    train["status"] = data["status"][:9000]
    train[score] =  data[score][:9000]
    
    test = {}
    test["status"] = data["status"][9001:]
    test[score] =  data[score][9001:]
    return pd.DataFrame.from_dict(train), pd.DataFrame.from_dict(test)
    
if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.ERROR)
    train_df, test_df = load_facebook_data()
    train_df.head()
    train_input_fn = tf.estimator.inputs.pandas_input_fn(
    train_df, train_df[score], num_epochs=None, shuffle=True)
    predict_train_input_fn = tf.estimator.inputs.pandas_input_fn(
    train_df, train_df[score], shuffle=False)
    predict_test_input_fn = tf.estimator.inputs.pandas_input_fn(
    test_df, test_df[score], shuffle=False)
    embedded_text_feature_column = hub.text_embedding_column(
    key="status", 
    module_spec="https://tfhub.dev/google/nnlm-en-dim128/1",)
    estimator = tf.estimator.DNNClassifier(
    hidden_units=[500, 100],
    feature_columns=[embedded_text_feature_column],
    n_classes=2,
    optimizer=tf.train.AdagradOptimizer(learning_rate=0.003,))
    estimator.train(input_fn=train_input_fn, steps=1000);
    train_eval_result = estimator.evaluate(input_fn=predict_train_input_fn)
    test_eval_result = estimator.evaluate(input_fn=predict_test_input_fn)
    print "Training set accuracy: {accuracy}".format(**train_eval_result)
    print "Test set accuracy: {accuracy}".format(**test_eval_result)