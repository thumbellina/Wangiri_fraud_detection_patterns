# Wangiri_fraud_detection_patterns
This repository contains the flask web application used to detect Wangiri fraud patterns detection from Call Detail Records(CDR) dataset. 
User guidelines :
1. Provide the dataset file name in tsv format in the filename parameter in flask_app_classification_patterns.py. Place the file within the main folder
2. Choose options for balancing data and cross validation in the initial parameters along with the pattern to be identified(Missed call(p1)/Callback(p2)/Broadcast(p3))
4. Run the file in flask environment
   - Give command EXPORT FLASK_APP="flask_app_classification_patterns.py"
   - flask run
5. In the application select the algorithm you would want to run
6. Results are visible in form of confusion matrix and accuracy, precision recall scores
7. Confusion matrix images are stored in static/images folder.

This repository created by Akshaya Ravi is part of the research  paper "Wangiri Fraud: Pattern Analysis and MachineLearning-based Detection":   https://ieeexplore.ieee.org/document/9772071
