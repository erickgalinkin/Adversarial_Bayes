from counterfit.core.targets import ArtTarget
import tensorflow as tf


class Mnist(ArtTarget):
    def __init__(self):
        self.clip_values = (0, 255)
        self.model = tf.keras.models.load_model(self.model_endpoint)
        (train_x, train_y), (test_x, test_y)  = tf.keras.datasets.mnist.load_data()
        self.X = train_x


    def __call__(self, x):
        return self.model.predict(x)

    model_name = "mnist_bayes"
    model_data_type = "numpy"
    model_input_shape = (28, 28)
    model_output_classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    model_endpoint = "/home/erick/dev/RobustMNIST/src/model/Experiment/"
    X = []
