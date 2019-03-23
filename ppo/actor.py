from keras.models import Model, load_model, save_model
from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, AveragePooling2D
from keras.activations import relu, softmax
from keras import backend as K
from keras.optimizers import Adam, RMSprop, SGD
import keras.initializers as initializers

class Actor():

    def __init__(self, state_size, action_size, hyper_param={}, seed=714):
        hyper_param = {
            'lr': 1e-6,
        }
        self.seed = 129

        self.state_size = state_size
        self.action_size = action_size
        
        state = Input(shape=self.state_size)
        advantage = Input(shape=(1, ))
        old_prediction = Input(shape=(self.action_size, ))

        x = Conv2D(filters=20, kernel_size=(2, 2), strides=1, activation=relu, padding='same')(state)
        x = AveragePooling2D()(x)
        x = Conv2D(filters=20, kernel_size=(4, 4), strides=1, activation=relu, padding='same')(x)
        x = AveragePooling2D()(x)
        x = Flatten()(x)
        x = Dense(units=512, activation=relu,
                kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=.00002, seed=self.seed),
                bias_initializer=initializers.Constant(0.1)
                )(x)
        actions_prob = Dense(units=action_size, activation=softmax,
                name='output')(x)

        model = Model(inputs=[state, advantage, old_prediction], outputs=actions_prob)
        model.compile(optimizer=
                        # SGD(lr=hyper_param['lr']),
                        # RMSprop(lr=hyper_param['lr']),
                        Adam(lr=hyper_param['lr']),
                        loss=[self.proximal_policy_optimization_loss(
                        advantage=advantage,
                        old_prediction=old_prediction)])
        
        model.summary()
        self.model = model

    def proximal_policy_optimization_loss(self, advantage, old_prediction):
        LOSS_CLIPPING = 0.2
        ENTROPY_LOSS = 0.01
        def loss(y_true, y_pred):
            prob = y_true * y_pred
            old_prob = y_true * old_prediction
            r = prob/(old_prob + 1e-10)
            return -K.mean(
                        K.minimum(
                            r * advantage, 
                            K.clip(
                                r, 
                                min_value=1 - LOSS_CLIPPING,
                                max_value=1 + LOSS_CLIPPING
                            ) * advantage
                        ) + ENTROPY_LOSS * (prob * K.log(prob + 1e-10))
                    )
        return loss

    def save_model(self, name):
        self.model.save(name)

    def load_model(self, name):
        # state = Input(shape=self.state_size)
        advantage = Input(shape=(1, ))
        old_prediction = Input(shape=(self.action_size, ))
        model = load_model(name, 
                        custom_objects={'loss': 
                            self.proximal_policy_optimization_loss(
                            advantage=advantage,
                            old_prediction=old_prediction)})
        self.model = model