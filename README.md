# Box punches detection machine learning project
The project was done as part of Pattern recognition & Machine learning Master course in UPC Barcelona in year 2017-18.

## Implementation
The project goal was to identify combinations of box punches. The data comes from two smartphones accelerometers. The implementation consists of extracting relevant features, training different classifiers with 6 different classes and testing classifiers on new data. The classes are:
* Jab (Left Straight) - class 1
* Cross (Right Straight) - class 2
* Left Hook - class 3
* Right Hook - class 4
* Left Uppercut - class 5
* Right Uppercut - class 6

## Features
Features used for each acceleration axis (6 axes) are:
* statistic - min, max, median, mean and standard deviation
* wavelet - 5 squared detail coefficients of Daubechies 3 wavelet. Libary used for obtaining wavelet features is PyWavelets.

## Files
* File cross_validation_comparison.py is used for cross-validation and determining the best classifier and features for our project.
* File testing_set_validation.py is used for testing the new data combinations with best classifier (K-nearest).

## Report
* Report of the project is located in the .pdf file