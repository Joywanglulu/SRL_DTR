import numpy as np
import math
# from keras.initializations import normal, identity
from keras.models import model_from_json, load_model
# from keras.engine.training import collect_trainable_weights
from keras.models import Sequential
from keras.layers import Dense, Flatten,Dot, Dropout,RepeatVector,Input,Masking, merge, Lambda,Reshape,LSTM,TimeDistributed,Embedding,concatenate,PReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam
import keras.backend as K
from keras.losses import mse
import tensorflow as tf

HIDDEN1_UNITS = 40
HIDDEN2_UNITS = 180
REWARD_THRESHOLD = 30
reg_lambda = 25
def avg(t,mask=None):
    if mask is None:
        return K.mean(t,-2)
    mask =  K.cast(mask,'float32')
    t = t*tf.expand_dims(mask,-1)
    t = K.sum(t,-2)/tf.expand_dims(K.sum(mask,-1),-1)
    return t

# def mse1(y_ture,y_pre):
#     reg_vector = tf.maximum(tf.abs(tf.reduce_sum(y_pre,axis=-1)) - REWARD_THRESHOLD, 0)
#     reg_term = tf.reduce_sum(reg_vector)
#     print('mse',mse(y_ture,y_pre))
#     print('reg_lambda', reg_lambda * reg_term)
#
#     return (mse(y_ture,y_pre) + reg_lambda * reg_term)+tf.reshape(tf.nn.l2_normalize(y_pre,dim=-1),(-1,self.time_stamp))

# def clipped_error(x):
#     try:
#         return tf.select(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)
#     except:
#         return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)
#
class CriticNetwork(object):
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE, epsilon, tiem_stamp, med_size, lab_size, demo_size, di_size, action_dim):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        self.epsilon = epsilon
        self.time_stamp = tiem_stamp
        self.med_size = med_size
        self.lab_size = lab_size
        self.demo_size = demo_size
        self.di_size = di_size
        self.action_dim = action_dim
        
        K.set_session(sess)
        self.min_delta = -1
        self.max_delta = 1

        #Now create the model
        self.model, self.weights,self.action, self.state,self.disease, self.demo = self.create_critic_network(state_size, action_size)
        self.sw = tf.placeholder(shape=[None, self.time_stamp, 1], dtype=tf.float32)
        self.target_model, self.target_weights,self.target_action, self.target_state,self.target_disease = self.create_critic_network(state_size, action_size)
        self.action_grads = tf.gradients((self.model.output), self.action)
        # self.reg_vector = tf.maximum(tf.abs(self.model.output) - REWARD_THRESHOLD, 0)
        # self.tar_q = tf.placeholder(shape=[None,self.time_stamp,1], dtype=tf.float32)
        #
        # self.tar_q = tf.multiply(self.tar_q,self.sw)
        # self.output = tf.multiply(self.model.output,self.sw)
        # self.delta = self.tar_q - self.output
        # self.clipped_delta = tf.clip_by_value(self.delta, self.min_delta, self.max_delta, name='clipped_delta')
        # self.global_step = tf.Variable(0, trainable=False)
        # print('self.clipped_delta',self.clipped_delta)
        #
        # self.loss = tf.reduce_mean((clipped_error(self.clipped_delta)), name='loss')
        # print(' self.loss',  self.loss)
        # params_grad1 = tf.gradients(self.loss, self.weights)
        # clip_gradient,_ = tf.clip_by_global_norm(params_grad1,1)
        # grads = zip(clip_gradient, self.weights)
        # self.optimize = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(grads)
        # self.sess.run(tf.initialize_all_variables())

    def gradients(self, states,disease,actions,sw):
        return self.sess.run(self.action_grads, feed_dict={
            self.state: states,self.disease: disease,
            self.action: actions,
            self.sw:np.reshape(sw,(-1,self.time_stamp,1))
        })[0]


    def train(self, states,disease,tar_q,action,sw):
      self.sess.run([self.optimize],feed_dict={
            self.state: states,
            self.disease: disease,
            self.tar_q: tar_q,
            self.action: action,
            self.sw: np.reshape(sw,(-1,self.time_stamp,1))
        })

    def target_train(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in range(len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU)* critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)

    def create_critic_network(self):
        print("Now we build the model")
        main_input_lab_test = Input(shape=(self.time_stamp, self.lab_size), dtype='float32')
        d1 = Dropout()(main_input_lab_test)
        main_input_demo = Input(shape=(self.demo_size), dtype='float32')
        demo = (Dense(HIDDEN1_UNITS))(main_input_demo)
        demo = PReLU()(demo)
        demo = RepeatVector(self.time_stamp)(demo)
        main_input_disease = Input(shape=(self.di_size,), dtype='int32')
        d2 = Dropout()(main_input_disease)
        e1 = (Embedding(output_dim=(HIDDEN1_UNITS), input_dim=(2001), input_length=self.time_stamp, mask_zero=True))(d2)
        emb_out = Lambda(avg)(e1)
        emb_out = RepeatVector(self.time_stamp)(emb_out)
        emb_out = TimeDistributed(Dense(HIDDEN1_UNITS))(emb_out)
        emb_out = PReLU()(emb_out)
        m1 = Masking(mask_value=0, input_shape=(self.time_stamp, self.lab_size))(d1)
        l1 = LSTM(
            batch_input_shape=(self.time_stamp, self.lab_size),
            output_dim = HIDDEN2_UNITS,
            return_sequences=True,
        )(m1)
        A = Input(shape=(self.time_stamp, self.action_dim),name='action2')
        # model_c1 = concatenate([l1, emb_out])
        model_c1 = merge([l1, emb_out, demo],mode='concat')
        a1 = TimeDistributed(Dense(HIDDEN2_UNITS, activation='linear'))(A)
        h2 =  merge([model_c1,a1],mode='sum')
        V = TimeDistributed(Dense(1,activation='linear'))(h2)
        model = Model(input=[main_input_lab_test, A, main_input_disease, main_input_demo],output=V)
        model.summary()
        adam = Adam(lr=self.LEARNING_RATE)
        model.compile(loss='mse', optimizer=adam, sample_weight_mode="temporal")

        return model, model.trainable_weights, A, main_input_lab_test, main_input_disease, main_input_demo
