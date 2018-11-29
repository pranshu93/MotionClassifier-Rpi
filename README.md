MotionClassifier-Rpi
============

This folder contains code for training an rnn model based on 3 Sensor readings(Accelerometer, Magnetometer & Gyroscope) being input at a certain frequency and classifies motion state into 3 classes(1. Stopped, 2. Moving Straight, 3. Turning). An already collected dataset has been deployed in "mod_data.csv". It further contains a predictor code designed to do a inference on RPi3 wherein it ups respective pin(GPIO 17, GPIO 27 and GPIO 22) corresponding to the prediction for the designated duration.

## Training

python rnn.py [Data_file] [Learning_rate] [#Epochs] [#Batch_size] [#Time Steps Dependency] [#Hidden Layer Size]

e.g. python rnn.py mod_data.csv 0.01 100 25 3 9

This will create a folder Parameters with appropriate trained weights and mean, std of distribution alongside #Time Step Dependency and #Hidden Layer Size in form of numpy arrays.

## Inference

python inference.py [#Frequency in number of time steps per second]

e.g. python inference.py 4

This code uses the weights created in Parameters folder to inference on Rpi3.
