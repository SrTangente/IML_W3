from scipy.io import arff
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder


def preprocess_vowel(data):
    vowel_df = pd.DataFrame(data)
    # Drop "train or test" and "speaker" columns
    vowel_df = vowel_df.drop("Train_or_Test", axis=1)
    vowel_df = vowel_df.drop("Speaker_Number", axis=1)
    # Encode classes to numbers
    enc = LabelEncoder()
    # Categorical encoding the categorical attributes
    vowel_df["Sex"] = enc.fit_transform(vowel_df["Sex"]).astype(float)
    vowel_df["Class"] = enc.fit_transform(vowel_df["Class"]).astype(float)
    # Normalize the data
    scaler = preprocessing.MinMaxScaler()
    vowel_df_scaled = scaler.fit_transform(vowel_df.values)
    vowel_df = pd.DataFrame(vowel_df_scaled)
    return vowel_df.to_numpy()


def read_vowel_fold(i):
    train_data, train_meta = arff.loadarff('./datasets/vowel/vowel.fold.00000'+i+'train.arff')
    test_data, test_meta = arff.loadarff('./datasets/vowel/vowel.fold.00000'+i+'test.arff')
    vowel_train = preprocess_vowel(train_data)
    vowel_test = preprocess_vowel(test_data)
    return vowel_train, vowel_test


def preprocess_adult(data):
    adult_df = pd.DataFrame(data)
    # Replace missing values
    adult_df = adult_df.replace(b"?", np.nan)
    # Handle missing values (for now, just delete the row with the missing value)
    # The index can be reset after dropna() by calling reset_index(drop=True)
    # The index reset is omitted for now in order not to lose track of the missing values dropped
    adult_df = adult_df.dropna()
    # Encode classes [b'<=50K', b'>50K'] to numbers
    enc = LabelEncoder()
    adult_df["class"] = enc.fit_transform(adult_df["class"]).astype(float)
    # One-hot encoding the categorical attributes
    adult_df = pd.get_dummies(adult_df)
    # Normalize the data
    scaler = preprocessing.MinMaxScaler()
    adult_df_scaled = scaler.fit_transform(adult_df.values)
    adult_df = pd.DataFrame(adult_df_scaled)
    return adult_df.to_numpy()


def read_adult_fold(i):
    train_data, train_meta = arff.loadarff('./datasets/adult/adult.fold.00000'+i+'train.arff')
    test_data, test_meta = arff.loadarff('./datasets/adult/adult.fold.00000'+i+'test.arff')
    adult_train = preprocess_adult(train_data)
    adult_test = preprocess_adult(test_data)
    return adult_train, adult_test