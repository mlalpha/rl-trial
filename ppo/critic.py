from keras.models import Model, load_model, save_model
from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, AveragePooling2D
from keras.activations import relu
from keras import backend as K
from keras.optimizers import Adam
import keras.initializers as initializers

class Critic():

    def __init__(self, state_size, action_size, hyper_param={}, seed=714):
        hyper_param = {
            'lr': 1e-8,
        }
        self.seed = 714

        state = Input(shape=state_size)
        
        x = Conv2D(filters=32, kernel_size=(8, 8), strides=4, activation=relu)(state)
        x = AveragePooling2D()(x)
        x = Conv2D(filters=20, kernel_size=(4, 4), strides=1, activation=relu, padding='same')(x)
        x = AveragePooling2D()(x)
        x = Flatten()(x)
        x = Dense(units=512, activation=relu,
                kernel_initializer=initializers.RandomNormal(mean=.0, stddev=.03, seed=self.seed),
                bias_initializer=initializers.Constant(0.1)
                )(x)
        q_values = Dense(units=1,
                bias_initializer='ones')(x)
 
        model = Model(inputs=state, outputs=q_values)
        model.compile(optimizer=Adam(lr=hyper_param['lr']),
              loss='mse',
              metrics=['accuracy'])

        model.summary()
        self.model = model

    def save_model(self, name):
        self.model.save(name)

    def load_model(self, name):
        self.model = load_model(name)