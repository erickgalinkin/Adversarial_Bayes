import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv1D, AveragePooling1D, BatchNormalization, MaxPooling1D, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
import numpy as np
import logging
import matplotlib.pyplot as plt

BATCH_SIZE = 128
EPOCHS = 25
LOGDIR = "./logs/"
MODEL_DIR = "./model/"

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# Simple LeNet-5 inspired model
def create_model(dropout=False, dropout_rate=0.5):
    input_shape = (28, 28)
    num_classes = 10

    inputs = tf.keras.Input(shape=input_shape)
    x = Conv1D(64, kernel_size=5, activation='tanh')(inputs)
    x = AveragePooling1D()(x)
    if dropout:
        x = Dropout(dropout_rate)(x)
    x = Conv1D(128, kernel_size=5, activation='tanh')(x)
    x = AveragePooling1D()(x)
    if dropout:
        x = Dropout(dropout_rate)(x)
    x = Conv1D(128, kernel_size=3, activation='tanh')(x)
    x = AveragePooling1D()(x)
    x = Flatten()(x)
    if dropout:
        x = Dropout(dropout_rate)(x)
    x = Dense(120, activation='tanh')(x)
    if dropout:
        x = Dropout(dropout_rate)(x)
    x = Dense(84, activation='tanh')(x)
    if dropout:
        x = Dropout(dropout_rate)(x)
    out = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=out)

    return model

# ResNet-18 inspired model
def residual_block(filters, repetitions, dropout, dropout_rate, first_layer=False):
    def block(inputs, filters, strides, dropout, dropout_rate, first_layer):
            if first_layer:
                conv = Conv1D(filters=filters, kernel_size=5, strides=strides, padding="same")(inputs)

            else:
                conv = brc_block(filters=filters, kernel_size=3, strides=strides)(inputs)

            if dropout:
                conv = Dropout(dropout_rate)(conv)

            residual = brc_block(filters=filters, kernel_size=3, strides=strides)(conv)
            if dropout:
                residual = Dropout(dropout_rate)(residual)
            return shortcut(inputs, residual)

    def fn(inputs):
        for i in range(repetitions):
            strides=1
            if i == 0 and not first_layer:
                strides = 2
            inputs = block(inputs, filters, strides, dropout, dropout_rate, first_layer)
        return inputs

    return fn


def bn_relu(inputs):
    bn = BatchNormalization()(inputs)
    return Activation("relu")(bn)


def cbr_block(**params):
    filters = params["filters"]
    kernel_size = params["kernel_size"]
    strides = params["strides"]

    def fn(inputs):
        conv = Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, padding="same",
                      kernel_regularizer='l2')(inputs)
        return bn_relu(conv)

    return fn


def brc_block(**params):
    filters = params["filters"]
    kernel_size = params["kernel_size"]
    strides = params["strides"]

    def fn(inputs):
        act = bn_relu(inputs)
        return Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, padding="same",
                      kernel_regularizer='l2')(act)

    return fn


def shortcut(inputs, residual):
    # If input and residual size match, we can just use the identity block here.
    if inputs.shape == residual.shape:
        return tf.keras.layers.Add()([inputs, residual])
    # Use convolution to fix dimensionality
    else:
        strides = int(inputs.shape[1]//residual.shape[1])
        shortcut = Conv1D(filters=residual.shape[-1], kernel_size=1, strides=strides, padding="valid")(inputs)
        return tf.keras.layers.Add()([shortcut, residual])


def create_resnet(reps=[2, 2, 2, 2], dropout=False, dropout_rate=0.4):
    # reps is a list of repetitions to use for the blocks.
    # The default will create a ResNet-18
    input_shape = (28, 28)
    num_classes = 10

    inputs = tf.keras.Input(shape=input_shape)
    conv1 = cbr_block(filters=32, kernel_size=5, strides=2)(inputs)
    pool1 = MaxPooling1D(padding="same")(conv1)
    if dropout:
        pool1 = Dropout(dropout_rate)(pool1)
    block = pool1
    filters = 2

    for r in reps:
        block = residual_block(filters=filters,
                               repetitions=r,
                               dropout=dropout,
                               dropout_rate=dropout_rate,
                               first_layer=(reps.index(r) == 0))(block)
        filters = filters * r + 1

    block = bn_relu(block)
    pool2 = AveragePooling1D()(block)
    if dropout:
        pool2 = Dropout(dropout_rate)(pool2)

    flatten = Flatten()(pool2)
    outputs = Dense(units=num_classes, activation="softmax")(flatten)

    model = Model(inputs=inputs, outputs=outputs)

    return model


if __name__ == "__main__":
    lr = 0.035

    opt = tf.keras.optimizers.SGD(learning_rate=lr)

    # Instantiate models for MNIST
    print("Instantiating models for MNIST...")
    control = create_model()
    experiment = create_model(dropout=True)
    control_resnet = create_resnet()
    experiment_resnet = create_resnet(dropout=True)

    control.compile(optimizer=opt, loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])
    experiment.compile(optimizer=opt, loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])
    control_resnet.compile(optimizer=opt, loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])
    experiment_resnet.compile(optimizer=opt, loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])

    modeldict = {"Control": control,
                 "Experiment": experiment,
                 "Control_ResNet": control_resnet,
                 "Experiment_ResNet": experiment_resnet
                 }

    resultsdict = dict()

    # Load MNIST data
    print("Loading MNIST data...")
    train, test = tf.keras.datasets.mnist.load_data()
    train_data, train_labels = train
    test_data, test_labels = test

    train_data = np.array(train_data, dtype=np.float32) / 255
    test_data = np.array(test_data, dtype=np.float32) / 255

    train_data = train_data.reshape(train_data.shape[0], 28, 28)
    test_data = test_data.reshape(test_data.shape[0], 28, 28)

    train_labels = np.array(train_labels, dtype=np.int32)
    test_labels = np.array(test_labels, dtype=np.int32)

    train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=10)
    test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=10)

    for name, model in modeldict.items():
        TBLOGDIR = LOGDIR + name
        tensorboard_callback = TensorBoard(log_dir=TBLOGDIR, histogram_freq=1)
        earlystopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=2)
        print(f'\n results for {name}')
        history = model.fit(train_data, train_labels,
                            epochs=EPOCHS,
                            validation_data=(test_data, test_labels),
                            batch_size=BATCH_SIZE,
                            callbacks=[tensorboard_callback, earlystopping])
        model.save(MODEL_DIR + name)
        resultsdict[name] = history

    # Create plot of training accuracy
    plt.style.use("ggplot")
    plt.figure(figsize=(12,9))
    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.set_xlim(1, 25)
    for label, hist in resultsdict.items():
        y = hist.history["val_accuracy"]
        x = np.linspace(1, len(y)+1, num=len(y))
        plt.plot(x, y, linewidth=3, label=label)
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('test set accuracy')
    plt.title("Model accuracy on test set")
    plt.savefig("test_accuracy.png")