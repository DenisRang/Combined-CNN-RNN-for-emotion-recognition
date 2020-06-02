import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input

from src import metrics
from src.processor import process_image


# from keras_vggface import utils


class Extractor():
    """
    Model wrapper class to perform extraction function from images to vectors using the CNN model
    """

    def __init__(self, saved_model):
        base_model = load_model(saved_model, custom_objects={'ccc_loss': metrics.ccc_loss,
                                                             'rmse': metrics.rmse,
                                                             'rmse_v': metrics.rmse_v,
                                                             'rmse_a': metrics.rmse_a,
                                                             'cc_v': metrics.cc_v,
                                                             'cc_a': metrics.cc_a,
                                                             'ccc_v': metrics.ccc_v,
                                                             'ccc_a': metrics.ccc_a})

        # We'll extract features at the dense layer with 300 units.
        self.extract_model = Model(
            inputs=base_model.input,
            outputs=base_model.get_layer('dense').output
        )
        input = Input((None, 300))
        output = base_model.get_layer('dense_1')(input)
        self.predict_model = Model(
            inputs=input,
            outputs=output
        )

    def extract(self, image_path):
        img = process_image(image_path)
        img = (img / 255.).astype(np.float32)
        return self.extract_image(img)

    def extract_image(self, img):
        x = np.expand_dims(img, axis=0)
        #         x = utils.preprocess_input(x, version=1)

        # Get the prediction.
        features = self.extract_model.predict(x)
        features = features[0]

        return features

    def predict(self, features):
        return self.predict_model.predict(features)
