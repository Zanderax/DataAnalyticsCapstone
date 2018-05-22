import tensorflow as tf
import tensorflow_hub as hub
# import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import shutil

# import seaborn as sns

import argparse

import csv
tempDir = os.environ['TMPDIR']

validScores = ["sNEU", "sAGR", "sCON", "sOPN", "sEXT" ]
validModules = ["https://tfhub.dev/google/nnlm-en-dim128/1"]

# Load all files from a directory in a DataFrame.
def load_facebook_data( score ):
    data = {}
    data["status"] = []
    data[score] = []
    with open('facebook_clean.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data["status"].append(row["STATUS"])
            data[score].append(int(float(row[score])) )
    train = {}
    train["status"] = data["status"][:9000]
    train[score] =  data[score][:9000]
    
    test = {}
    test["status"] = data["status"][9001:]
    test[score] =  data[score][9001:]
    return pd.DataFrame.from_dict(train), pd.DataFrame.from_dict(test), test


def RunModel( score, module ):
    # We only care about errors
    tf.logging.set_verbosity(tf.logging.ERROR)
    
    # Set up our data frames
    train_df, test_df, test = load_facebook_data(score)
    train_df.head()
    
    #
    train_input_fn = tf.estimator.inputs.pandas_input_fn(
    train_df, train_df[score], num_epochs=None, shuffle=True)
    predict_train_input_fn = tf.estimator.inputs.pandas_input_fn(
    train_df, train_df[score], shuffle=False)
    predict_test_input_fn = tf.estimator.inputs.pandas_input_fn(
    test_df, test_df[score], shuffle=False)
    
    # Set up our feature columns from the model
    embedded_text_feature_column = hub.text_embedding_column(
        key="status", 
        module_spec=module,)
        
    # Set up our neural network 
    estimator = tf.estimator.DNNClassifier(
        hidden_units=[500, 100],
        feature_columns=[embedded_text_feature_column],
        n_classes=6,
        optimizer=tf.train.AdagradOptimizer(learning_rate=0.003,))
        
    # Train our neural network
    estimator.train(input_fn=train_input_fn, steps=10)
    
    # Get Results
    train_eval_result = estimator.evaluate(input_fn=predict_train_input_fn)
    test_eval_result = estimator.evaluate(input_fn=predict_test_input_fn)
    test_predictions = estimator.predict(input_fn=predict_test_input_fn)
    
    train_accuracy = train_eval_result["accuracy"] * 100.
    test_accuracy = test_eval_result["accuracy"] * 100.
    
    
    # Print results
    print "Results for " + score
    print "================"
    print "Training set accuracy: {0:.2f}%".format(train_accuracy)
    print "Test set accuracy: {0:.2f}%".format(test_accuracy)
    print ""
    
def ParseArgs():
    parser = argparse.ArgumentParser(description='Model Arguments')
    parser.add_argument('--scores', help='The score to use', choices=validScores, default=validScores[0], nargs='*')
    parser.add_argument('--module', help='The module to use', choices=validModules, default=validModules[0])
    return parser.parse_args()
    
def ClearTmpDir():
    fileList = os.listdir(tempDir)
    for fileName in fileList:
        shutil.rmtree(tempDir+"/"+fileName)
    
def main():
    args = ParseArgs()
    scores = args.scores
    module = args.module
    ClearTmpDir()
    for score in scores:
        RunModel( score, module )
        ClearTmpDir()
       
    
if __name__ == "__main__":
    main()
    