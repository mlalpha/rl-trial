from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization
from keras.activations import relu, softmax
from keras import backend as K
from keras.optimizers import Adam

class Actor():

    def __init__(self, state_size, action_size, hyper_param={}, seed=714):
        hyper_param = {
            'lr': 1e-6,
        }

        
        state = Input(shape=(84,84,1, ))
        advantage = Input(shape=(1, ))
        old_prediction = Input(shape=(action_size, ))
        
        x = Conv2D(filters=32, kernel_size=(8, 8), strides=4, activation=relu)(state)
        x = BatchNormalization()(x)
        x = Conv2D(filters=64, kernel_size=(4, 4), strides=2, activation=relu)(x)
        x = BatchNormalization()(x)
        x = Conv2D(filters=64, kernel_size=(3, 3), strides=1, activation=relu)(x)
        x = BatchNormalization()(x)
        x = Flatten()(x)
        x = Dense(units=512, activation=relu,
                kernel_initializer='random_uniform',
                bias_initializer='zeros')(x)
        actions_prob = Dense(units=action_size, activation=softmax,
                kernel_initializer='random_uniform',
                bias_initializer='zeros', name='output')(x)

        model = Model(inputs=[state, advantage, old_prediction], outputs=actions_prob)
        model.compile(optimizer=Adam(lr=hyper_param['lr']),
              loss=[self.proximal_policy_optimization_loss(
                          advantage=advantage,
                          old_prediction=old_prediction)])
        
        model.summary()
        self.model = model

    def proximal_policy_optimization_loss(self, advantage, old_prediction):
        LOSS_CLIPPING = 0.2
        ENTROPY_LOSS = 1e-3
        def loss(y_true, y_pred):
            prob = y_true * y_pred
            old_prob = y_true * old_prediction
            r = prob/(old_prob + 1e-10)
            return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantage) + ENTROPY_LOSS * (prob * K.log(prob + 1e-10)))
        return loss


