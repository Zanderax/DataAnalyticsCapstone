import argparse
import csv
import numpy as np
import os
import pandas as pd
import re
import shutil
import tensorflow as tf
import tensorflow_hub as hub

#import matplotlib.pyplot as plt
#import seaborn as sns

tempDir = os.environ['TMPDIR']

validScores = ["sNEU", "sAGR", "sCON", "sOPN", "sEXT" ]
validModules = {
    "nnlm-en-dim128":"https://tfhub.dev/google/nnlm-en-dim128/1",
    "random-nnlm-en-dim128":"https://tfhub.dev/google/random-nnlm-en-dim128/1",
}

# Load all files from a directory in a DataFrame.
def load_facebook_data( score ):
    data = {}
    data["status"] = []
    data[score] = []
    with open('facebook_clean.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data["status"].append(row["STATUS"])
            data[score].append(float(row[score]) )
    train = {}
    train["status"] = data["status"][:9000]
    train[score] =  data[score][:9000]
    
    test = {}
    test["status"] = data["status"][9001:]
    test[score] =  data[score][9001:]
    return pd.DataFrame.from_dict(train), pd.DataFrame.from_dict(test), test
    
# def get_predictions(estimator, input_fn):
#     return [x["class_ids"][0] for x in estimator.predict(input_fn=input_fn)]



def RunClassificationModel( score, module, train_module=False ):
    # We only care about errors
    tf.logging.set_verbosity(tf.logging.ERROR)
    
    # Set up our data frames
    train_df, test_df, test = load_facebook_data(score)
    train_df.head()
    
    # Train model
    train_input_fn = tf.estimator.inputs.pandas_input_fn(
        train_df, train_df[score], num_epochs=None, shuffle=True)
    predict_train_input_fn = tf.estimator.inputs.pandas_input_fn(
        train_df, train_df[score], shuffle=False)
    predict_test_input_fn = tf.estimator.inputs.pandas_input_fn(
        test_df, test_df[score], shuffle=False)
    
    # Set up our feature columns from the model
    embedded_text_feature_column = hub.text_embedding_column(
        key="status", 
        module_spec=module,
        trainable=train_module,)
        
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
    print "Results"
    print "================"
    print "Model: " + module
    print "Score: " + score
    print "Train Module: " + str(train_module)
    print "================"
    print "Training set accuracy: {0:.2f}%".format(train_accuracy)
    print "Test set accuracy: {0:.2f}%".format(test_accuracy)
    print ""

def RunRegressionModel( score, module, train_module=False ):
    # We only care about errors
    tf.logging.set_verbosity(tf.logging.ERROR)
    
    # Set up our data frames
    train_df, test_df, test = load_facebook_data(score)
    train_df.head()
    
    # Train model
    train_input_fn = tf.estimator.inputs.pandas_input_fn(
        train_df, train_df[score], num_epochs=None, shuffle=True)
    predict_train_input_fn = tf.estimator.inputs.pandas_input_fn(
        train_df, train_df[score], shuffle=False)
    predict_test_input_fn = tf.estimator.inputs.pandas_input_fn(
        test_df, test_df[score], shuffle=False)
    
    # Set up our feature columns from the model
    embedded_text_feature_column = hub.text_embedding_column(
        key="status", 
        module_spec=module,
        trainable=train_module,)
        
    # Set up our neural network 
    estimator = tf.estimator.DNNRegressor(
        feature_columns=[embedded_text_feature_column],
        hidden_units=[1024, 512, 256],
        optimizer=tf.train.ProximalAdagradOptimizer(
          learning_rate=0.1,
          l1_regularization_strength=0.001
        ))
        
    # Train our neural network
    estimator.train(input_fn=train_input_fn, steps=100)

    ev = estimator.evaluate(input_fn=predict_train_input_fn)
    
    predictor = estimator.predict(input_fn=predict_test_input_fn)
    predictions = list(predictor)
    
    labels = test_df[score].tolist()
    
    predictionList = []
    
    for prediction in predictions:
        predictionList.append(prediction["predictions"][0].item())


    predictionsTensor = tf.convert_to_tensor(predictionList)
        
    labelsTensor = tf.convert_to_tensor(labels)
    
    
    rmse, rmse_op  = tf.metrics.root_mean_squared_error(labelsTensor,predictionsTensor)
    
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()
    # print(sess.run([rmse, rmse_op]))
    # print(sess.run([rmse]))


    
    # Print results
    print "Results"
    print "================"
    print "Model: " + module
    print "Score: " + score
    print "Train Module: " + str(train_module)
    print "RMSE: " + str(sess.run([rmse, rmse_op])[1])
    print "================"
    # # print "Training set accuracy: {0:.2f}%".format(train_accuracy)
    # print "Test set accuracy: {0:.2f}%".format(test_accuracy)
    # print ""
    


def ParseArgs():
    parser = argparse.ArgumentParser(description='Model Arguments')
    parser.add_argument('--scores', help='The score to use', choices=validScores, default=validScores[0], nargs='*')
    parser.add_argument('--module', help='The module to use', choices=validModules.keys(), default=validModules.keys()[0])
    parser.add_argument('--train', dest='train', action='store_true')
    parser.add_argument('--regression', dest='regression', action='store_true')
    parser.set_defaults(train=False)
    parser.set_defaults(regression=False)
    return parser.parse_args()
    
def ClearTmpDir():
    fileList = os.listdir(tempDir)
    for fileName in fileList:
        shutil.rmtree(tempDir+"/"+fileName)
    
def main():
    args = ParseArgs()
    scores = args.scores
    module = args.module
    train = args.train
    regression = args.regression
    ClearTmpDir()
    for score in scores:
        if regression:
            RunRegressionModel( score, validModules[module], train )
        else:
            RunClassificationModel( score, validModules[module], train )
        ClearTmpDir()
       
    
if __name__ == "__main__":
    main()
    