"""
A collection of models we'll use to attempt to predict valence and arousal from videos.
"""
import sys

# from keras_squeeze_excite_network.se_resnet import SEResNet50
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, AvgPool2D, GRU
# from keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import SGD

from src import metrics


# from keras_vggface.vggface import VGGFace
# import keras_vggface


class ResearchModels():

    def __init__(self, model, saved_model=None):

        # Get the appropriate model.
        if saved_model is not None:
            print("Loading model %s" % saved_model)
            model = load_model(saved_model, custom_objects={'ccc_loss': metrics.ccc_loss,
                                                            'rmse': metrics.rmse,
                                                            'rmse_v': metrics.rmse_v,
                                                            'rmse_a': metrics.rmse_a,
                                                            'cc_v': metrics.cc_v,
                                                            'cc_a': metrics.cc_a,
                                                            'ccc_v': metrics.ccc_v,
                                                            'ccc_a': metrics.ccc_a}, compile=False)

            model.trainable = True
            for layer in model.layers:
                layer.trainable = True
                if hasattr(layer, 'layers'):
                    for nested_layer in layer.layers:
                        nested_layer.trainable = True
            opt = SGD(lr=.01, decay=1e-5, momentum=.9)
            model.compile(loss=metrics.ccc_loss,
                          optimizer=opt,
                          metrics=[metrics.rmse,
                                   metrics.rmse_v,
                                   metrics.rmse_a,
                                   metrics.cc_v,
                                   metrics.cc_a,
                                   metrics.ccc_v,
                                   metrics.ccc_a])
            self.model = model
        # elif model == 'simple_cnn':
        #     print("Loading Simple CNN by Khorrami.")
        #     self.input_shape = (96, 96, 1)
        #     self.model = self.simple_cnn()
        elif model == 'simple_cnn_ccc_loss':
            print("Loading Simple CNN by Khorrami with CCC loss")
            self.input_shape = (96, 96, 1)
            self.model = self.simple_cnn_ccc_loss()
        # elif model == 'simple-cnn-l1-0.01':
        #     print("Loading Simple CNN by Khorrami with l1 normalization 0.01.")
        #     self.input_shape = (96, 96, 1)
        #     self.model = self.simple_cnn_l1(0.01)
        # elif model == 'simple-cnn-l2-0.01':
        #     print("Loading Simple CNN by Khorrami with l2 normalization 0.01.")
        #     self.input_shape = (96, 96, 1)
        #     self.model = self.simple_cnn_l2(0.01)
        # elif model == 'simple-cnn-l2-0.005':
        #     print("Loading Simple CNN by Khorrami with l2 normalization 0.005.")
        #     self.input_shape = (96, 96, 1)
        #     self.model = self.simple_cnn_l2(0.005)
        # elif model == 'senet_pret_adam':
        #     print("Loading SeResNet model pretrained on imagenet with Adam optimizer.")
        #     self.input_shape = (112, 112, 3)
        #     self.model = self.senet_pret_adam()
        # elif model == 'senet_pret_adam_ccc_loss':
        #     print("Loading SeResNet model pretrained on imagenet with Adam optimizer and CCC loss.")
        #     self.input_shape = (112, 112, 3)
        #     self.model = self.senet_pret_adam_ccc_loss()
        # elif model == 'vggface_ccc_loss':
        #     print("Loading VGG16 model pretrained on FaceNet with CCC loss.")
        #     self.input_shape = (112, 112, 3)
        #     self.model = self.vggface_ccc_loss()
        # elif model == 'vggface_2dropout_ccc_loss':
        #     print("Loading VGG16 model pretrained on FaceNet with 2 dropout blocks and CCC loss.")
        #     self.input_shape = (112, 112, 3)
        #     self.model = self.vggface_2dropout_ccc_loss()
        # elif model == 'simple_rnn':
        #     print("Loading SimpleRNN model.")
        #     self.model = self.simple_rnn()
        # elif model == 'simple_rnn_ccc_loss':
        #     print("Loading SimpleRNN model with CCC loss.")
        #     self.model = self.simple_rnn_ccc_loss()
        # elif model == 'simple_rnn_tanh_3_layers':
        #     print("Loading SimpleRNN model with 3 layers and tanh activation.")
        #     self.model = self.simple_rnn_tanh_3_layers()
        # elif model == 'simple_rnn_tanh_3_layers_ccc_loss':
        #     print("Loading SimpleRNN model with 3 layers and tanh activation and ccc loss.")
        #     self.model = self.simple_rnn_tanh_3_layers_ccc_loss()
        # elif model == 'simple_rnn_relu':
        #     print("Loading SimpleRNN model with ReLU activation.")
        #     self.model = self.simple_rnn_relu()
        # elif model == 'simple_rnn_relu_3_layers':
        #     print("Loading SimpleRNN model with 3 layers, ReLU activation.")
        #     self.model = self.simple_rnn_relu_3_layers()
        # elif model == 'lstm':
        #     print("Loading LSTM model.")
        #     self.model = self.lstm()
        # elif model == 'gru':
        #     print("Loading GRU model.")
        #     self.model = self.gru()
        # elif model == 'gru_1_layer_ccc_loss':
        #     print("Loading 1 layer GRU model.")
        #     self.model = self.gru_1_layer_ccc_loss()
        elif model == 'gru_ccc_loss':
            print("Loading GRU model with CCC loss.")
            self.model = self.gru_ccc_loss()
        else:
            print("Unknown network.")
            sys.exit()
        print(self.model.summary())

    # def simple_cnn_l2(self, weight_decay):
    #     model = Sequential()
    #     model.add(BatchNormalization(input_shape=self.input_shape))
    #     model.add(Conv2D(64,
    #                      (5, 5),
    #                      padding='same',
    #                      activation='relu',
    #                      use_bias=False, kernel_regularizer=l2(weight_decay)))
    #     model.add(MaxPooling2D(pool_size=(2, 2)))
    #     model.add(Conv2D(128,
    #                      (5, 5),
    #                      activation='relu',
    #                      use_bias=False, kernel_regularizer=l2(weight_decay)))
    #     model.add(MaxPooling2D(pool_size=(2, 2)))
    #     model.add(Conv2D(256,
    #                      (5, 5),
    #                      activation='relu',
    #                      use_bias=False, kernel_regularizer=l2(weight_decay)))
    #     model.add(AvgPool2D(pool_size=(9, 9),
    #                         strides=9))
    #     model.add(Flatten())
    #     model.add(Dense(300, kernel_regularizer=l2(weight_decay)))
    #     model.add(Dropout(0.5))
    #     model.add(Dense(2))
    #
    #     opt = SGD(lr=.01, decay=1e-5, momentum=.9)
    #     model.compile(loss='mean_squared_error',
    #                   optimizer=opt,
    #                   metrics=[metrics.rmse,
    #                            metrics.rmse_v,
    #                            metrics.rmse_a,
    #                            metrics.cc_v,
    #                            metrics.cc_a,
    #                            metrics.ccc_v,
    #                            metrics.ccc_a])
    #     return model
    #
    # def simple_cnn_l1(self, weight_decay):
    #     model = Sequential()
    #     model.add(BatchNormalization(input_shape=self.input_shape))
    #     model.add(Conv2D(64,
    #                      (5, 5),
    #                      padding='same',
    #                      activation='relu',
    #                      use_bias=False, kernel_regularizer=l1(weight_decay)))
    #     model.add(MaxPooling2D(pool_size=(2, 2)))
    #     model.add(Conv2D(128,
    #                      (5, 5),
    #                      activation='relu',
    #                      use_bias=False, kernel_regularizer=l1(weight_decay)))
    #     model.add(MaxPooling2D(pool_size=(2, 2)))
    #     model.add(Conv2D(256,
    #                      (5, 5),
    #                      activation='relu',
    #                      use_bias=False, kernel_regularizer=l1(weight_decay)))
    #     model.add(AvgPool2D(pool_size=(9, 9),
    #                         strides=9))
    #     model.add(Flatten())
    #     model.add(Dense(300, kernel_regularizer=l1(weight_decay)))
    #     model.add(Dropout(0.5))
    #     model.add(Dense(2))
    #
    #     opt = SGD(lr=.01, decay=1e-5, momentum=.9)
    #     model.compile(loss='mean_squared_error',
    #                   optimizer=opt,
    #                   metrics=[metrics.rmse,
    #                            metrics.rmse_v,
    #                            metrics.rmse_a,
    #                            metrics.cc_v,
    #                            metrics.cc_a,
    #                            metrics.ccc_v,
    #                            metrics.ccc_a])
    #     return model
    #
    # def simple_cnn(self):
    #     model = Sequential()
    #     model.add(BatchNormalization(input_shape=self.input_shape))
    #     model.add(Conv2D(64,
    #                      (5, 5),
    #                      padding='same',
    #                      activation='relu',
    #                      use_bias=False))
    #     model.add(MaxPooling2D(pool_size=(2, 2)))
    #     model.add(Conv2D(128,
    #                      (5, 5),
    #                      activation='relu',
    #                      use_bias=False))
    #     model.add(MaxPooling2D(pool_size=(2, 2)))
    #     model.add(Conv2D(256,
    #                      (5, 5),
    #                      activation='relu',
    #                      use_bias=False))
    #     model.add(AvgPool2D(pool_size=(9, 9),
    #                         strides=9))
    #     model.add(Flatten())
    #     model.add(Dense(300))
    #     model.add(Dropout(0.5))
    #     model.add(Dense(2))
    #
    #     opt = SGD(lr=.01, decay=1e-5, momentum=.9)
    #     model.compile(loss='mean_squared_error',
    #                   optimizer=opt,
    #                   metrics=[metrics.rmse,
    #                            metrics.rmse_v,
    #                            metrics.rmse_a,
    #                            metrics.cc_v,
    #                            metrics.cc_a,
    #                            metrics.ccc_v,
    #                            metrics.ccc_a])
    #     # model.compile(loss='mean_squared_error',
    #     #               optimizer=opt,
    #     #               metrics=[metrics.rmse,
    #     #                        metrics.cc,
    #     #                        metrics.ccc])
    #     return model

    def simple_cnn_ccc_loss(self):
        model = Sequential()
        model.add(BatchNormalization(input_shape=self.input_shape))
        model.add(Conv2D(64,
                         (5, 5),
                         padding='same',
                         activation='relu',
                         use_bias=False))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128,
                         (5, 5),
                         activation='relu',
                         use_bias=False))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(256,
                         (5, 5),
                         activation='relu',
                         use_bias=False))
        model.add(AvgPool2D(pool_size=(9, 9),
                            strides=9))
        model.add(Flatten())
        model.add(Dense(300))
        model.add(Dropout(0.5))
        model.add(Dense(2))

        opt = SGD(lr=.01, decay=1e-5, momentum=.9)
        model.compile(loss=metrics.ccc_loss,
                      optimizer=opt,
                      metrics=[metrics.rmse,
                               metrics.rmse_v,
                               metrics.rmse_a,
                               metrics.cc_v,
                               metrics.cc_a,
                               metrics.ccc_v,
                               metrics.ccc_a])
        return model

    # def vggface_ccc_loss(self):
    #     hidden_dim = 300
    #
    #     vgg_model = VGGFace(include_top=False, model='vgg16', input_shape=(112, 112, 3))
    #     for layer in vgg_model.layers:
    #         layer.trainable = False
    #     last_layer = vgg_model.get_layer('pool5').output
    #     x = Flatten(name='flatten')(last_layer)
    #     x = Dense(hidden_dim, name='fc6')(x)
    #     x = Dropout(0.5)(x)
    #     out = Dense(2, name='fc8')(x)
    #     model = Model(vgg_model.input, out)
    #
    #     opt = SGD(lr=.01, decay=1e-5, momentum=.9)
    #     model.compile(loss=metrics.ccc_loss,
    #                   optimizer=opt,
    #                   metrics=[metrics.rmse,
    #                            metrics.rmse_v,
    #                            metrics.rmse_a,
    #                            metrics.cc_v,
    #                            metrics.cc_a,
    #                            metrics.ccc_v,
    #                            metrics.ccc_a])
    #     return model
    #
    # def vggface_2dropout_ccc_loss(self):
    #
    #     base_model = VGGFace(include_top=False, model='vgg16', input_shape=(112, 112, 3))
    #     #         for layer in base_model.layers:
    #     #             layer.trainable = False
    #     last_layer = base_model.get_layer('pool5').output
    #
    #     x = Dropout(0.5)(last_layer)
    #     x = Conv2D(512,
    #                (1, 1),
    #                activation=keras.layers.LeakyReLU(alpha=0.3))(x)
    #     x = AvgPool2D(pool_size=(3, 3), strides=1)(x)
    #     av_pool = Flatten()(x)
    #
    #     x = Dropout(0.5)(last_layer)
    #     x = Conv2D(512,
    #                (1, 1),
    #                activation=keras.layers.LeakyReLU(alpha=0.3))(x)
    #     x = MaxPooling2D(pool_size=(3, 3), strides=1)(x)
    #     max_pool = Flatten()(x)
    #
    #     concat = Concatenate()([av_pool, max_pool])
    #     output = Dense(2)(concat)
    #     model = keras.models.Model(inputs=[base_model.input], outputs=[output])
    #
    #     opt = SGD(lr=.01, decay=1e-5, momentum=.9)
    #     model.compile(loss=metrics.ccc_loss,
    #                   optimizer=opt,
    #                   metrics=[metrics.rmse,
    #                            metrics.rmse_v,
    #                            metrics.rmse_a,
    #                            metrics.cc_v,
    #                            metrics.cc_a,
    #                            metrics.ccc_v,
    #                            metrics.ccc_a])
    #     return model
    #
    # def senet_pret_adam(self):
    #     base_model = SEResNet50(input_shape=self.input_shape, include_top=False, weights='imagenet')
    #
    #     x = Dropout(0.5)(base_model.output)
    #     x = Conv2D(512,
    #                (1, 1),
    #                activation=tf.keras.layers.LeakyReLU(alpha=0.3))(x)
    #     x = AvgPool2D(pool_size=(4, 4), strides=1)(x)
    #     av_pool = Flatten()(x)
    #
    #     x = Dropout(0.5)(base_model.output)
    #     x = Conv2D(512,
    #                (1, 1),
    #                activation=tf.keras.layers.LeakyReLU(alpha=0.3))(x)
    #     x = MaxPooling2D(pool_size=(4, 4), strides=1)(x)
    #     max_pool = Flatten()(x)
    #
    #     concat = Concatenate()([av_pool, max_pool])
    #     output = Dense(2)(concat)
    #     model = tf.keras.models.Model(inputs=[base_model.input], outputs=[output])
    #
    #     # train
    #     # model.compile(optimizer='SGD', loss='mean_squared_error', metrics=['accuracy'])
    #     # opt = SGD(lr=.01, decay=1e-5, momentum=.9)
    #     opt = Adam(lr=.001, decay=1e-3)
    #     model.compile(loss='mean_squared_error',
    #                   # model.compile(loss=root_mean_squared_error,
    #                   optimizer=opt,
    #                   metrics=[metrics.rmse,
    #                            metrics.rmse_v,
    #                            metrics.rmse_a,
    #                            metrics.cc_v,
    #                            metrics.cc_a,
    #                            metrics.ccc_v,
    #                            metrics.ccc_a])
    #     return model
    #
    # def senet_pret_adam_ccc_loss(self):
    #     base_model = SEResNet50(input_shape=self.input_shape, include_top=False, weights='imagenet')
    #
    #     x = Dropout(0.5)(base_model.output)
    #     x = Conv2D(512,
    #                (1, 1),
    #                activation=tf.keras.layers.LeakyReLU(alpha=0.3))(x)
    #     x = AvgPool2D(pool_size=(4, 4), strides=1)(x)
    #     av_pool = Flatten()(x)
    #
    #     x = Dropout(0.5)(base_model.output)
    #     x = Conv2D(512,
    #                (1, 1),
    #                activation=tf.keras.layers.LeakyReLU(alpha=0.3))(x)
    #     x = MaxPooling2D(pool_size=(4, 4), strides=1)(x)
    #     max_pool = Flatten()(x)
    #
    #     concat = Concatenate()([av_pool, max_pool])
    #     output = Dense(2)(concat)
    #     model = tf.keras.models.Model(inputs=[base_model.input], outputs=[output])
    #
    #     opt = Adam(lr=.001, decay=1e-3)
    #     model.compile(loss=metrics.ccc_loss,
    #                   optimizer=opt,
    #                   metrics=[metrics.rmse,
    #                            metrics.rmse_v,
    #                            metrics.rmse_a,
    #                            metrics.cc_v,
    #                            metrics.cc_a,
    #                            metrics.ccc_v,
    #                            metrics.ccc_a])
    #     return model
    #
    # def simple_rnn(self):
    #     model = Sequential()
    #     model.add(SimpleRNN(100, return_sequences=True, input_shape=[None, 300]))
    #     model.add(TimeDistributed(Dense(2)))
    #     opt = SGD(lr=.01, clipvalue=100, momentum=.9)
    #     model.compile(loss='mean_squared_error',
    #                   optimizer=opt,
    #                   metrics=[metrics.rmse,
    #                            metrics.rmse_v,
    #                            metrics.rmse_a,
    #                            metrics.cc_v,
    #                            metrics.cc_a,
    #                            metrics.ccc_v,
    #                            metrics.ccc_a])
    #     return model
    #
    # def simple_rnn_ccc_loss(self):
    #     model = Sequential()
    #     model.add(SimpleRNN(100, return_sequences=False, input_shape=[100, 300]))
    #     model.add(Dense(2))
    #     opt = SGD(lr=.01, clipvalue=100, momentum=.9)
    #     model.compile(loss=metrics.ccc_loss,
    #                   optimizer=opt,
    #                   metrics=[metrics.rmse,
    #                            metrics.rmse_v,
    #                            metrics.rmse_a,
    #                            metrics.cc_v,
    #                            metrics.cc_a,
    #                            metrics.ccc_v,
    #                            metrics.ccc_a])
    #     return model
    #
    # def simple_rnn_tanh_3_layers(self):
    #     model = Sequential()
    #     model.add(SimpleRNN(100, return_sequences=True, input_shape=[None, 300]))
    #     model.add(SimpleRNN(100, return_sequences=True))
    #     model.add(SimpleRNN(50, return_sequences=True))
    #     model.add(TimeDistributed(Dense(2)))
    #     opt = SGD(lr=.01, clipvalue=50, momentum=.9)
    #     model.compile(loss='mean_squared_error',
    #                   optimizer=opt,
    #                   metrics=[metrics.rmse,
    #                            metrics.rmse_v,
    #                            metrics.rmse_a,
    #                            metrics.cc_v,
    #                            metrics.cc_a,
    #                            metrics.ccc_v,
    #                            metrics.ccc_a])
    #     return model
    #
    # def simple_rnn_tanh_3_layers_ccc_loss(self):
    #     model = Sequential()
    #     model.add(SimpleRNN(100, return_sequences=True, input_shape=[None, 300]))
    #     model.add(SimpleRNN(100, return_sequences=True))
    #     model.add(SimpleRNN(50, return_sequences=True))
    #     model.add(TimeDistributed(Dense(2)))
    #     opt = SGD(lr=.01, momentum=.9)
    #     model.compile(loss=metrics.ccc_loss,
    #                   optimizer=opt,
    #                   metrics=[metrics.rmse,
    #                            metrics.rmse_v,
    #                            metrics.rmse_a,
    #                            metrics.cc_v,
    #                            metrics.cc_a,
    #                            metrics.ccc_v,
    #                            metrics.ccc_a])
    #     return model
    #
    # def simple_rnn_relu(self):
    #     model = Sequential()
    #     model.add(SimpleRNN(100, activation='relu', return_sequences=True, input_shape=[None, 300]))
    #     model.add(TimeDistributed(Dense(2)))
    #     opt = SGD(lr=.01, clipvalue=10, momentum=.9)
    #     model.compile(loss='mean_squared_error',
    #                   optimizer=opt,
    #                   metrics=[metrics.rmse,
    #                            metrics.rmse_v,
    #                            metrics.rmse_a,
    #                            metrics.cc_v,
    #                            metrics.cc_a,
    #                            metrics.ccc_v,
    #                            metrics.ccc_a])
    #     return model
    #
    # def simple_rnn_relu_3_layers(self):
    #     model = Sequential()
    #     model.add(SimpleRNN(100, activation='relu', return_sequences=True, input_shape=[None, 300]))
    #     model.add(SimpleRNN(100, activation='relu', return_sequences=True))
    #     model.add(SimpleRNN(50, activation='relu', return_sequences=True))
    #     model.add(TimeDistributed(Dense(2)))
    #     opt = SGD(lr=.01, clipvalue=10, momentum=.9)
    #     model.compile(loss='mean_squared_error',
    #                   optimizer=opt,
    #                   metrics=[metrics.rmse,
    #                            metrics.rmse_v,
    #                            metrics.rmse_a,
    #                            metrics.cc_v,
    #                            metrics.cc_a,
    #                            metrics.ccc_v,
    #                            metrics.ccc_a])
    #     return model
    #
    # def lstm(self):
    #     model = Sequential()
    #     model.add(LSTM(100, return_sequences=True, input_shape=[None, 300]))
    #     model.add(LSTM(100, return_sequences=True))
    #     model.add(LSTM(50, return_sequences=True))
    #     model.add(TimeDistributed(Dense(2)))
    #     opt = SGD(lr=.01, clipvalue=10, momentum=.9)
    #     model.compile(loss=metrics.ccc_loss,
    #                   optimizer=opt,
    #                   metrics=[metrics.rmse,
    #                            metrics.rmse_v,
    #                            metrics.rmse_a,
    #                            metrics.cc_v,
    #                            metrics.cc_a,
    #                            metrics.ccc_v,
    #                            metrics.ccc_a])
    #
    #     return model
    #
    # def gru(self):
    #     model = Sequential()
    #     model.add(GRU(100, return_sequences=True, input_shape=[None, 300]))
    #     model.add(GRU(100, return_sequences=True))
    #     model.add(GRU(50, return_sequences=True))
    #     model.add(TimeDistributed(Dense(2)))
    #     opt = SGD(lr=.01, decay=1e-5, momentum=.9)
    #     model.compile(loss='mean_squared_error',
    #                   optimizer=opt,
    #                   metrics=[metrics.rmse,
    #                            metrics.rmse_v,
    #                            metrics.rmse_a,
    #                            metrics.cc_v,
    #                            metrics.cc_a,
    #                            metrics.ccc_v,
    #                            metrics.ccc_a])
    #
    #     return model

    def gru_ccc_loss(self):
        model = Sequential()
        model.add(GRU(100, return_sequences=True, input_shape=[100, 300]))
        model.add(GRU(100, return_sequences=True))
        model.add(GRU(50, return_sequences=False))
        model.add(Dense(2))
        # lr_schedule = ExponentialDecay(
        #     initial_learning_rate=1e-2,
        #     decay_steps=400,
        #     decay_rate=0.97)
        # opt = Adam(learning_rate=lr_schedule)
        opt = SGD(lr=.01, decay=1e-5, momentum=.9)
        model.compile(loss=metrics.ccc_loss,
                      optimizer=opt,
                      metrics=[metrics.rmse,
                               metrics.rmse_v,
                               metrics.rmse_a,
                               metrics.cc_v,
                               metrics.cc_a,
                               metrics.ccc_v,
                               metrics.ccc_a])

        return model

    # def gru_1_layer_ccc_loss(self):
    #     model = Sequential()
    #     model.add(GRU(100, return_sequences=False, input_shape=[100, 300]))
    #     model.add(Dense(2))
    #     opt = SGD(lr=.01, decay=1e-5, momentum=.9)
    #     model.compile(loss=metrics.ccc_loss,
    #                   optimizer=opt,
    #                   metrics=[metrics.rmse,
    #                            metrics.rmse_v,
    #                            metrics.rmse_a,
    #                            metrics.cc_v,
    #                            metrics.cc_a,
    #                            metrics.ccc_v,
    #                            metrics.ccc_a])
    #
    #     return model


# class FinalCnnRnnModel():
#     def __init__(self, cnn_model_path=SAVED_CNN_EXTRACTOR_MODEL, rnn_model_path=SAVED_RNN_MODEL):
#         cnn_model = load_model(cnn_model_path, custom_objects={'rmse': metrics.rmse,
#                                                                'rmse_v': metrics.rmse_v,
#                                                                'rmse_a': metrics.rmse_a,
#                                                                'cc_v': metrics.cc_v,
#                                                                'cc_a': metrics.cc_a,
#                                                                'ccc_v': metrics.ccc_v,
#                                                                'ccc_a': metrics.ccc_a})
#         rnn_model = load_model(rnn_model_path, custom_objects={'ccc_loss': metrics.ccc_loss,
#                                                                'rmse': metrics.rmse,
#                                                                'rmse_v': metrics.rmse_v,
#                                                                'rmse_a': metrics.rmse_a,
#                                                                'cc_v': metrics.cc_v,
#                                                                'cc_a': metrics.cc_a,
#                                                                'ccc_v': metrics.ccc_v,
#                                                                'ccc_a': metrics.ccc_a})
#
#         extractor = Model(
#             inputs=cnn_model.input,
#             outputs=cnn_model.get_layer('dense').output
#         )
#
#         model = Sequential()
#         model.add(Input(shape=[None, 96, 96, 1]))
#         model.add(TimeDistributed(extractor))
#         model.add(rnn_model)
#         self.model = model
