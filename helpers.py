import librosa
import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import time
import random

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import (
    Dense,
    Conv1D,
    Flatten,
    MaxPooling1D,
    Dropout,
    Conv2D,
    MaxPooling2D,
)
from tensorflow.keras.applications import VGG16
from tensorflow.keras import Model

# pre-defined classes present in data
classifications = [
    "Angry",
    "Defence",
    "Fighting",
    "Happy",
    "HuntingMind",
    "Mating",
    "MotherCall",
    "Paining",
    "Resting",
    "Warning",
]


# check if file exists, load if so
def check_file(file, path):
    # if the data has been loaded and pickled, open it
    if file in os.listdir(path):
        print("Data found, loading contents...")
        with open(path + file, "rb") as f:
            data = pickle.load(f)
            return data

    print("Data not found.")
    return None


# load raw data
def load_data(meow_path):
    data = []

    for classification in classifications:
        meow_classification = os.path.join(meow_path, classification)

        # load files
        for filename in os.listdir(meow_classification):
            if "_aug" not in filename and filename.endswith(".mp3"):
                try:
                    audio, sr = librosa.load(
                        os.path.join(meow_classification, filename)
                    )

                    # append sample data and classification to the list as a dictionary
                    data.append({"classification": classification, "audio": audio})

                except Exception as e:
                    print(f"Error loading {filename}: {e}")
                    continue

        # notify at end of each folder
        print(f"Loaded {classification} files")

    return data


# function for data augmentation
def augment(data):
    start_time = time.time()

    # Add noise
    noise = np.random.randn(len(data))
    augmented_noise = data + 0.00001 * noise
    print(f"Added noise in {time.time() - start_time}")

    # Shift pitch
    start_time = time.time()
    augmented_pitch = []
    for wave in data:
        augmented_pitch.append(
            librosa.effects.pitch_shift(wave, sr=22050, n_steps=np.random.randint(0, 5))
        )
    print(f"Augmented pitch in {time.time() - start_time}")

    # Combine as one dataset
    return pd.concat((data, augmented_noise, pd.Series(augmented_pitch)))


# return augmented data
def get_augmented_data(data):
    aug_X = augment(data["audio"])
    max_length = aug_X.apply(len).max()

    aug_y = pd.concat(
        [data["classification"], data["classification"], data["classification"]]
    )

    z_length = len(data["classification"])
    aug_z = pd.concat(
        [
            pd.Series(np.zeros(z_length)),
            pd.Series(np.ones(z_length)),
            pd.Series(np.ones(z_length)),
        ]
    )

    aug_data = pd.DataFrame(
        {"audio": aug_X, "classification": aug_y, "augmented": aug_z}
    )

    aug_data = aug_data.sample(frac=1).reset_index(drop=True)

    return aug_data, max_length


def grab_data():
    # path to data in gdrive
    meow_path = "/content/drive/MyDrive/cs109b/MeowData/NAYA_DATA_AUG1X"

    # check if data exists and load if not
    data = check_file("data.pkl", "./data/")
    if data is not None:
        return data

    # load data if file is not found
    print("Loading data...")
    start_time = time.time()
    data = pd.DataFrame(load_data(meow_path))
    print(f"Loaded data in {time.time() - start_time}")

    # begin augmenting data
    print("Augmenting...")
    start_time = time.time()
    aug_data, max_length = get_augmented_data(data)
    print(f"Augmented data in {time.time() - start_time}")
    print("Cleaning and pickling...")
    start_time = time.time()

    # pickle the augmented data
    with open("./data/aug_data_dirty.pkl", "wb") as f:
        pickle.dump(aug_data, f)

    # clean the augmented data
    def clean_data(audio, max_size=370368):
        audio = np.trim_zeros(audio, "f")
        audio = np.pad(audio, (0, max_size - len(audio)))
        audio = audio / np.max(np.abs(audio))
        return audio

    print("Cleaning...")
    start_time = time.time()
    aug_data["audio"] = aug_data["audio"].apply(clean_data)
    print(f"Cleaned data in {time.time() - start_time}")

    return aug_data


def sample(aug_data, aug=False, num_samples=25):
    """
    Generates a sample of the dataset.

    Args:
    - aug: Whether or not augmented data is included
    - num_samples: Number of data samples per classification

    Returns:
    - sample_X_train
    - sample_X_test
    - sample_y_train
    - sample_y_test

    """
    max_len = aug_data["audio"].apply(len).max()

    data = pd.DataFrame(aug_data)

    # drop augmented files if aug=False, otherwise skip this step
    if aug == False:
        data = data[data["augmented"] != 1]

    def clean_data(audio, max_size=max_len):
        audio = np.trim_zeros(audio, "f")
        audio = np.pad(audio, (0, max_size - len(audio)))
        audio = audio / np.max(np.abs(audio))
        return audio

    # shuffle and sample
    sample_data = data.sample(frac=1, random_state=109)

    # group data by classification
    grouped_data = sample_data.groupby("classification")

    # sample an equal number of samples from each group
    sampled_dfs = []
    for _, group in grouped_data:
        sampled_dfs.append(group.sample(n=num_samples, replace=False, random_state=109))

    # concatenate the sampled dataframes and split into X and y
    balanced_sample_df = pd.concat(sampled_dfs)
    sample_X = balanced_sample_df["audio"]
    sample_y = balanced_sample_df["classification"]

    # split the balanced sample dataframe into balanced train and test sets
    sample_X_train, sample_X_test, sample_y_train, sample_y_test = train_test_split(
        sample_X,
        sample_y,
        test_size=0.2,
        stratify=balanced_sample_df["classification"],
        random_state=109,
    )

    return sample_X_train, sample_X_test, sample_y_train, sample_y_test


# sampling rate
sr = 22050


# create the melspectrogram and return magnitudes
def mel_spectrogram(audio, hop_length_mult=4):
    # number of mel frequency bands (frequency axis will be divided into these bands which are 1 mel apart)
    n_mels = 128

    # create mel spectrogram with assigned multiple of n_mels as hop length
    S = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=n_mels, hop_length=n_mels * hop_length_mult
    )
    log_S = librosa.amplitude_to_db(S)
    return log_S


# transform the data
def mel_transform(X, y, hop_length_mult):
    # create spectrograms for each audio
    print("creating spectrograms")
    s = time.time()
    X = X.apply(lambda x: mel_spectrogram(x, hop_length_mult))
    print(f"{time.time()-s} ms")

    # one hot encode the 10 classes
    print("hot encoding")
    s = time.time()
    y = OneHotEncoder(sparse_output=False).fit_transform(
        LabelEncoder().fit_transform(y).reshape(-1, 1)
    )
    print(f"{time.time()-s} ms")

    # convert the data to a tensor and add an extra dimension
    print("converting to tensor")
    s = time.time()
    arrays = np.stack(X.values)
    arrays = arrays[..., np.newaxis]
    X = tf.convert_to_tensor(arrays, dtype=tf.float32)
    print(f"{time.time()-s} ms")

    return X, y


# create and return the model for predicting on mel spectrograms
def mel_model(input_shape=(128, 724, 1)):
    # set the default input shape
    input_shape = input_shape

    # create same model (2d equivalent) as ft model
    model = Sequential(
        [
            Conv2D(
                filters=32, kernel_size=3, activation="relu", input_shape=input_shape
            ),
            MaxPooling2D(pool_size=2),
            Conv2D(filters=64, kernel_size=3, activation="relu"),
            MaxPooling2D(pool_size=2),
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.5),
            Dense(10, activation="softmax", name="prediction_layer"),
        ]
    )

    model.compile(
        optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"]
    )

    model.summary()

    return model


def plot_performance(model_histories, hops=True):
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    model_labels = ["Baseline", "Baseline VGG", "Regularized VGG", "Best"]
    hop_lengths = [512, 256, 128]

    # Training Accuracy
    sns.set(style="darkgrid")
    for i, history in enumerate(model_histories):
        if hops:
            label = hop_lengths[i]

            sns.lineplot(
                ax=ax[0][0],
                x=range(1, 4),
                y=history.history["accuracy"],
                label=f"Hop Length: {label}",
            )
        else:
            sns.lineplot(
                ax=ax[0][0],
                x=range(1, 11),
                y=history["accuracy"],
                label=f"{model_labels[i]}",
            )

    ax[0][0].set(xlabel="Epoch", ylabel="Accuracy Score")
    ax[0][0].set(title=f"Models' Training Accuracies vs Epochs")
    ax[0][0].legend()
    if hops:
        ax[0][0].set_xticks(range(1, 4))
    else:
        ax[0][0].set_xticks(range(1, 11))

    # Validation Accuracy
    for i, history in enumerate(model_histories):
        if hops:
            label = hop_lengths[i]

            sns.lineplot(
                ax=ax[0][1],
                x=range(1, 4),
                y=history.history["val_accuracy"],
                label=f"Hop Length: {label}",
            )
        else:
            sns.lineplot(
                ax=ax[0][1],
                x=range(1, 11),
                y=history["val_accuracy"],
                label=f"{model_labels[i]}",
            )

    ax[0][1].set(xlabel="Epoch", ylabel="Validation Accuracy Score")
    ax[0][1].set(title=f"Models' Validation Accuracies vs Epochs")
    ax[0][1].legend()
    if hops:
        ax[0][1].set_xticks(range(1, 4))
    else:
        ax[0][1].set_xticks(range(1, 11))

    # Training Loss
    for i, history in enumerate(model_histories):
        if hops:
            label = hop_lengths[i]

            sns.lineplot(
                ax=ax[1][0],
                x=range(1, 4),
                y=history.history["loss"],
                label=f"Hop Length: {label}",
            )
        else:
            sns.lineplot(
                ax=ax[1][0],
                x=range(1, 11),
                y=history["loss"],
                label=f"{model_labels[i]}",
            )
    ax[1][0].set(xlabel="Epoch", ylabel="Loss")
    ax[1][0].set(title=f"Models' Training Loss vs Epochs")
    ax[1][0].legend()
    if hops:
        ax[1][0].set_xticks(range(1, 4))
    else:
        ax[1][0].set_xticks(range(1, 11))

    # Validation Loss
    for i, history in enumerate(model_histories):
        if hops:
            label = hop_lengths[i]

            sns.lineplot(
                ax=ax[1][1],
                x=range(1, 4),
                y=history.history["val_loss"],
                label=f"Hop Length: {label}",
            )
        else:
            sns.lineplot(
                ax=ax[1][1],
                x=range(1, 11),
                y=history["val_loss"],
                label=f"{model_labels[i]}",
            )

    ax[1][1].set(xlabel="Epoch", ylabel="Validation Loss")
    ax[1][1].set(title=f"Models' Validation Loss vs Epochs")
    ax[1][1].legend()
    if hops:
        ax[1][1].set_xticks(range(1, 4))
    else:
        ax[1][1].set_xticks(range(1, 11))

    plt.tight_layout()


def vgg(nodes=512, dropout=0.5):
    # transfer learn using vgg16
    vgg = VGG16(weights="imagenet", include_top=False, input_shape=(128, 724, 3))

    # turn off training
    for layer in vgg.layers:
        layer.trainable = False

    # add a few classifier layers
    x = Flatten()(vgg.output)
    x = Dense(nodes, activation="relu")(x)
    x = Dropout(dropout)(x)
    output = Dense(10, activation="softmax")(
        x
    )  # num_classes is the number of classes in your dataset

    # create model
    model = Model(inputs=vgg.input, outputs=output)

    model.compile(
        optimizer=Adam(), loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    model.summary()
    return model


def compute_saliency_map(model, input_spec):
    """
    Computes the saliency map for a given CNN model and input spectrogram.

    Args:
    - model: Pre-trained CNN model
    - input_spec: Input audio spectrogram

    Returns:
    - saliency_map: Saliency map highlighting important regions of the input spectrogram
    """
    input_tensor = tf.convert_to_tensor(input_spec[np.newaxis, ...], dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(input_tensor)
        predictions = model(input_tensor)
        top_class = tf.argmax(predictions, axis=1)
        top_score = predictions[0, top_class.numpy()[0]]

    gradients = tape.gradient(top_score, input_tensor)
    saliency_map = tf.reduce_max(tf.abs(gradients), axis=-1)
    return saliency_map.numpy()
