### Azure ML modelling

This is a simple project where a classification model to predict credit card default is built using Azure ML.
-  Classification model used is Sklearn GradientBoostingClassifier.
-  Here the Azure ML resources like compute, models, endpoints, experiments are used.
-  MLFlow is used to track the experiments and the model is deployed using Azure ML endpoints.

1. src contains the main model building code.
2. Deploy contains a sample test data to test the deployed model.
3. dependencies contains the environment dependencies file.
4. The python notebook contains the code to spwan the compute cluster, train the model, register the model and deploy the model.

