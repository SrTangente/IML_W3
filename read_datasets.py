from scipy.io import arff
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder


def preprocess_vowel(train_data, test_data):
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)
    n = len(train_df)
    vowel_df = pd.concat([train_df, test_df])
    # Drop "train or test" and "speaker" columns
    vowel_df = vowel_df.drop("Train_or_Test", axis=1)
    vowel_df = vowel_df.drop("Speaker_Number", axis=1)
    # Encode classes to numbers
    enc = LabelEncoder()
    # Categorical encoding the categorical attributes
    vowel_df["Sex"] = enc.fit_transform(vowel_df["Sex"]).astype(float)
    vowel_df["Class"] = enc.fit_transform(vowel_df["Class"]).astype(float)
    # Save and remove label target
    y_train = vowel_df["Class"][:n].to_numpy()
    y_test = vowel_df["Class"][n:].to_numpy()
    vowel_df.drop("Class", axis=1)
    # Normalize the data
    scaler = preprocessing.MinMaxScaler()
    vowel_df_scaled = scaler.fit_transform(vowel_df.values)
    vowel_df = pd.DataFrame(vowel_df_scaled)
    vowel_df = vowel_df.to_numpy()
    return vowel_df[:n], y_train, vowel_df[n:], y_test


def read_vowel_fold(i):
    train_data, train_meta = arff.loadarff('./datasets/vowel/vowel.fold.00000'+str(i)+'.train.arff')
    test_data, test_meta = arff.loadarff('./datasets/vowel/vowel.fold.00000'+str(i)+'.test.arff')
    return preprocess_vowel(train_data, test_data)


def preprocess_adult(train_data, test_data):
    test_df = pd.DataFrame(test_data)
    train_df = pd.DataFrame(train_data)
    # Replace missing values
    train_df = train_df.replace(b"?", np.nan)
    test_df = test_df.replace(b"?", np.nan)
    # Handle missing values (for now, just delete the row with the missing value)
    # The index can be reset after dropna() by calling reset_index(drop=True)
    # The index reset is omitted for now in order not to lose track of the missing values dropped
    train_df = train_df.dropna()
    test_df = test_df.dropna()
    # Encode classes [b'<=50K', b'>50K'] to numbers
    enc = LabelEncoder()
    train_df["class"] = enc.fit_transform(train_df["class"]).astype(float)
    test_df["class"] = enc.fit_transform(test_df["class"]).astype(float)
    # Save and remove label target
    y_train = train_df["class"].to_numpy()
    y_test = test_df["class"].to_numpy()
    train_df.drop("class", axis=1)
    test_df.drop("class", axis=1)
    # One-hot encoding the categorical attributes
    n = len(train_df)
    adult_df = pd.concat([train_df, test_df])
    adult_df = pd.get_dummies(adult_df)
    # Normalize the data
    scaler = preprocessing.MinMaxScaler()
    adult_df_scaled = scaler.fit_transform(adult_df.values)
    adult_df = pd.DataFrame(adult_df_scaled)
    adult_np = adult_df.to_numpy()
    return adult_np[:n], y_train, adult_np[n:], y_test


def read_adult_fold(i):
    train_data, train_meta = arff.loadarff('./datasets/adult/adult.fold.00000'+str(i)+'.train.arff')
    test_data, test_meta = arff.loadarff('./datasets/adult/adult.fold.00000'+str(i)+'.test.arff')
    return preprocess_adult(train_data, test_data)

