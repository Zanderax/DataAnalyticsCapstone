import tensorflow as tf
import tensorflow_hub as hub
# import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
# import seaborn as sns

import csv

score = "sNEU"

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
            data[score].append(int(float(row[score])) )
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
    return pd.DataFrame.from_dict(train), pd.DataFrame.from_dict(test), test
    
if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.ERROR)
    train_df, test_df, test = load_facebook_data()
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
    n_classes=6,
    optimizer=tf.train.AdagradOptimizer(learning_rate=0.003,))
    estimator.train(input_fn=train_input_fn, steps=10);
    train_eval_result = estimator.evaluate(input_fn=predict_train_input_fn)
    test_eval_result = estimator.evaluate(input_fn=predict_test_input_fn)
    test_predictions = estimator.predict(input_fn=predict_test_input_fn)
    #print "Test type = " + str(type(test))
    print "test score type = " + str(type(test[score]))
    # print "test_predictions dir = " + str(dir(test_predictions))
    # print "gi_code type = " + str(type(test_predictions.gi_code))
    # print "gi_frame type = " + str(type(test_predictions.gi_frame))
    # print "gi_running type = " + str(type(test_predictions.gi_running))
    #rmse = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(test[score], test_predictions))))
    scoreTensor = tf.convert_to_tensor( test[score])
    # mse = tf.metrics.mean_squared_error(scoreTensor, test_predictions.gi_running )
    
    #tf.metrics.mean_squared_error(
    print "Training set accuracy: {accuracy}".format(**train_eval_result)
    print "Test set accuracy: {accuracy}".format(**test_eval_result)
    #print "RMSE = {rmse}".format(rmse)
    #print "MSE Type = {mse}".format(type(mse))