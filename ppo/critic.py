from keras.models import Model, Sequential
from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization
from keras.activations import relu
from keras import backend as K
from keras.optimizers import Adam
import keras.initializers as initializers

class Critic():

    def __init__(self, state_size, action_size, hyper_param={}, seed=714):
        hyper_param = {
            'lr': 1e-6,
        }
        self.seed = 714

        state = Input(shape=state_size)
        
        x = Conv2D(filters=32, kernel_size=(8, 8), strides=4, activation=relu)(state)
        x = BatchNormalization()(x)
        x = Conv2D(filters=64, kernel_size=(4, 4), strides=2, activation=relu)(x)
        x = BatchNormalization()(x)
        x = Conv2D(filters=64, kernel_size=(3, 3), strides=1, activation=relu)(x)
        x = Flatten()(x)
        x = Dense(units=512, activation=relu,
                kernel_initializer=initializers.RandomUniform(seed=self.seed),
                bias_initializer='zeros'
                )(x)
        q_values = Dense(units=1,
                bias_initializer='zeros')(x)
 
        model = Model(inputs=state, outputs=q_values)
        model.compile(optimizer=Adam(lr=hyper_param['lr']),
              loss='mse',
              metrics=['accuracy'])

        model.summary()
        self.model = model

