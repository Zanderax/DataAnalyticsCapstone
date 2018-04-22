import tensorflow as tf
import tensorflow_hub as hub
# import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
# import seaborn as sns

import csv

# Load all files from a directory in a DataFrame.
def load_facebook_data():
    data = {}
    data["status"] = []
    data["ext"] = []
    # data["neu"] = []
    # data["agr"] = []
    # data["con"] = []
    # data["opn"] = []
    with open('facebook_clean.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data["status"].append(row["STATUS"])
            ext = float(row["sEXT"])
            if ext < 2.5:
                data["ext"].append(0)
            else:
                data["ext"].append(1)
            # data["neu"].append(row["sNEU"])
            # data["agr"].append(row["sAGR"])
            # data["con"].append(row["sCON"])
            # data["opn"].append(row["sOPN"])\
    train = {}
    train["status"] = data["status"][:9000]
    train["ext"] =  data["ext"][:9000]
    
    test = {}
    test["status"] = data["status"][9001:]
    test["ext"] =  data["ext"][9001:]
    return pd.DataFrame.from_dict(train), pd.DataFrame.from_dict(test)
    
if __name__ == "__main__":
    print 0
    tf.logging.set_verbosity(tf.logging.ERROR)
    print 1
    train_df, test_df = load_facebook_data()
    print 2
    train_df.head()
    print 3
    train_input_fn = tf.estimator.inputs.pandas_input_fn(
    train_df, train_df["ext"], num_epochs=None, shuffle=True)
    print 4
    
    predict_train_input_fn = tf.estimator.inputs.pandas_input_fn(
    train_df, train_df["ext"], shuffle=False)
    print 5
    predict_test_input_fn = tf.estimator.inputs.pandas_input_fn(
    test_df, test_df["ext"], shuffle=False)
    print 6
    embedded_text_feature_column = hub.text_embedding_column(
    key="status", 
    module_spec="https://tfhub.dev/google/nnlm-en-dim128/1",)
    print 7
    estimator = tf.estimator.DNNClassifier(
    hidden_units=[500, 100],
    feature_columns=[embedded_text_feature_column],
    n_classes=2,
    optimizer=tf.train.AdagradOptimizer(learning_rate=0.003))
    print 8
    estimator.train(input_fn=train_input_fn, steps=1000);
    print 9
    train_eval_result = estimator.evaluate(input_fn=predict_train_input_fn)
    print 10
    test_eval_result = estimator.evaluate(input_fn=predict_test_input_fn)
    print 11
    
    print "Training set accuracy: {accuracy}".format(**train_eval_result)
    print "Test set accuracy: {accuracy}".format(**test_eval_result)