
## Predicting the sale price of Bulldozers using machine learning

IMPORTANT INFORMATION: The model was too large to upload. You can generate it follwing the steps in this project and drop it in the "trained_models" folder witht eh name "prediction_model.pkl". 

In this notebook, we're going to go through an example machine learning project with the goal of predicting the sale price of bulldozers.

#### Problem definition 

How well can we predict the future sale price of a bulldozer, given its characteristics and previous examples of how much similar bulldozers have been sold for?

#### Data
 The data is downloaded from the Kaggle Bluebook for Bulldozers competition: https://www.kaggle.com/c/bluebook-for-bulldozers/data

There are 3 main datasets:

Train.csv is the training set, which contains data through the end of 2011.

Valid.csv is the validation set, which contains data from January 1, 2012 - April 30, 2012 You make predictions on this set throughout the majority of the competition. Your score on this set is used to create the public leaderboard.

Test.csv is the test set, which won't be released until the last week of the competition. It contains data from May 1, 2012 - November 2012. Your score on the test set determines your final rank for the competition.

#### Evaluation

The evaluation metric for this competition is the RMSLE (root mean squared log error) between the actual and predicted auction prices.
For more on the evaluation of this project check: https://www.kaggle.com/c/bluebook-for-bulldozers/overview/evaluation

Note: The goal for most regression evaluation metrics is to minimize the error. For example, our goal for this project will be to build a machine learning model which minimises RMSLE.

Features Kaggle provides a data dictionary detailing all of the features of the dataset. You can view this data dictionary on Google Sheets: https://docs.google.com/spreadsheets/d/18ly-bLR8sbDJLITkWG7ozKm8l3RyieQ2Fpgix-beSYI/edit?usp=sharing

## Running Locally

Add PYTHONPATH variable for `~/.bash_profile ` for MacOS


## Virtual Environment
Install virtualenv

```python
python3 -m pip install virtualenv
```

Check version
```python
virtualenv --version
```

Create virtual environment

```python
virtualenv ml_package
```

Activate virtual environment

For Linux/Mac
```python
source ml_package/bin/activate
```
For Windows
```python
ml_package\Scripts\activate
```

Deactivate virtual environment

```python
deactivate
```


## Directory structure

```bash
prediction_model


├── MANIFEST.in
├── prediction_model
│   ├── config
│   │   ├── config.py
│   │   └── __init__.py
│   ├── datasets
│   │   ├── __init__.py
│   │   ├── test.csv
│   │   └── train.csv
│   ├── __init__.py
│   ├── pipeline.py
│   ├── predict.py
│   ├── processing
│   │   ├── data_handling.py
│   │   ├── __init__.py
│   │   └── preprocessing.py
│   ├── trained_models
│   │   ├── classification.pkl
│   │   └── __init__.py
│   ├── training_pipeline.py
│   └── VERSION
├── README.md
├── requirements.txt
├── setup.py
└── tests
    ├── pytest.ini
    └── test_prediction.py
```


# Build the Package

1. Goto Project directory and install dependencies
`pip install -r requirements.txt`

2. Create Pickle file after training:
`python prediction_model/training_pipeline.py`

3. Create source distribution and wheel
`python setup.py sdist bdist_wheel`

# Installation of Package

Go to project directory where `setup.py` file is located

1. To install it in editable or developer mode
```python
pip install -e .
```
```.``` refers to current directory

```-e``` refers to --editable mode

2. Normal installation
```python
pip install .
```
```.``` refers to current directory

3. Also can be installed from git as well after pushing to github

```
pip install git+https://github.com/charliepaks/bulldozer-prediction.git
```

# Testing the Package Working

1. Remove the PYTHONPATH from environment variables 
2. Goto a separate location which is outside of package directory
3. Create a new virual environment using the commands mentioned above & activate it
4. Before installing, test whether you are able to import the package of `prediction_model` - (you should not be able to do it)
5. Now in the new environment install the package using the generated file
`pip install git+https://github.com/charliepaks/bulldozer-prediction.git`
6. Now try importing the prediction_model, you should be able to do it successfully
7. Extras : Run training pipeline using the package, and also conduct the test


