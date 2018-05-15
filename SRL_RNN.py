import numpy as np
import random
import argparse
from keras.models import model_from_json, Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from sklearn.metrics import roc_auc_score, \
                label_ranking_average_precision_score, label_ranking_loss, jaccard_similarity_score

import tensorflow as tf
# from keras.engine.training import collect_trainable_weights
import json
import pandas as pd
from ActorNetwork1 import ActorNetwork
from CriticNetwork import CriticNetwork
from OU import OU
from keras import backend as K
import copy

df = pd.read_csv('/Users/Downloads/unreal-master/model/train_all_12_31_scale.csv')
val_df = pd.read_csv('/Users/Downloads/unreal-master/model/val_all_12_31_scale.csv')

df_disease = pd.read_csv('/Users/PycharmProjects/mimic_dataprocess/for_ij/train_di_base.csv')
val_df_di = pd.read_csv('/Users/PycharmProjects/mimic_dataprocess/for_ij/val_di_base.csv')

df_sta = pd.read_csv('/Users/PycharmProjects/mimic_dataprocess/for_ij/train_stastic_12_23.csv')
val_sta = pd.read_csv('/Users/PycharmProjects/mimic_dataprocess/for_ij/val_stastic_12_23.csv')

val_df_disease = val_df_di.drop_duplicates()
val_df_disease=val_df_disease.drop('hadm_id',axis=1)
val_df_disease =val_df_disease.values
val_df_di = val_df_di.drop('hadm_id',axis=1)
val_df_di=val_df_di.values
val_sta = val_sta.values

med_size = 180 #1000
di_size = 39
di = []
for i in range(di_size):
    di.append(str(i))
ac = []
for i in range(med_size):
    ac.append('l'+str(i))

demo = ['sofa','GENDER','RELIGION','MARITAL_STATUS','age','weight','height','language','ethnicity']
lab_test = ['dbp','fio2','GCS','blood_glucose','sbp','hr','PH','rr','bos','temp','urine_output']

df['prob'] = abs(df['flag'])
unique_id =df['hadm_id'].drop_duplicates().values
OU = OU()

class SRL_RNN:

    def __init__(self,config):
        self.config = config
        np.random.seed(config.model_seed)


    def batch(batch_size):

        state_size = 12
        states = None
        meds = None
        rewards = None
        next_states = None
        done_flags = None
        disease = []
        demos = []

        for bat in range(batch_size):
            traj_id = np.random.choice(unique_id)
            a = df.loc[df['hadm_id'] == traj_id]
            di = df_disease[di][df_disease['hadm_id'] == traj_id].values
            demo = df_sta[demo][df_sta['hadm_id'] == traj_id].values
            x = 0

            for i in a.index:
                disease.append(di)
                demos.append(demo)
                x = x + 1
                state = a.ix[i, lab_test]
                state = np.reshape(state, [1, state_size])
                med = a.ix[i, ac]
                med = np.reshape(med, [1, med_size])
                reward = a.ix[i, 'flag']
                if x < len(a):
                    med = med
                    next_state = df.ix[i + 1, lab_test]
                    reward = reward
                    done = 0
                    next_state = np.reshape(next_state, [1, state_size])
                else:
                    med = med
                    next_state = np.zeros(state_size)
                    reward = reward
                    done = 1
                    next_state = np.reshape(next_state, [1, state_size])

                if states is None:
                    states = copy.deepcopy(state)
                else:
                    states = np.vstack((states, state))
                if meds is None:
                    meds = copy.deepcopy(med)
                else:
                    meds = np.vstack((meds, med))
                if rewards is None:
                    rewards = [reward]
                else:
                    rewards = np.vstack((rewards, reward))
                if next_states is None:
                    next_states = copy.deepcopy(next_state)
                else:
                    next_states = np.vstack((next_states, next_state))
                if done_flags is None:
                    done_flags = [done]
                else:
                    done_flags = np.vstack((done_flags, done))

        return (states, np.squeeze(meds), np.squeeze(rewards), next_states, np.squeeze(done_flags), np.squeeze(np.array(disease)), np.squeeze(np.array(demos)))


    def process_batch(size,weight):

        a = df.sample(n=size,weights=weight )
        states = None
        meds = None
        rewards = None
        next_states = None
        done_flags = None

        for i in a.index:
            cur_state = a.ix[i, lab_test]
            med = a.ix[i, ac].values
            med = np.reshape(med, med_size)
            reward = a.ix[i, 'flag']

            if i != df.index[-1]:
                # if not terminal step in trajectory
                if df.ix[i, 'hadm_id'] == df.ix[i + 1, 'hadm_id']:
                    next_state = df.ix[i + 1, lab_test]
                    done = 0
                else:
                    next_state = np.zeros(len(cur_state))
                    done = 1
            else:
                next_state = np.zeros(len(cur_state))
                done = 1

            if states is None:
                states = copy.deepcopy(cur_state)
            else:
                states = np.vstack((states, cur_state))

            if meds is None:
                meds = [med]
            else:
                meds = np.vstack((meds, med))

            if rewards is None:
                rewards = [reward]
            else:
                rewards = np.vstack((rewards, reward))

            if next_states is None:
                next_states = copy.deepcopy(next_state)
            else:
                next_states = np.vstack((next_states, next_state))

            if done_flags is None:
                done_flags = [done]
            else:
                done_flags = np.vstack((done_flags, done))

        return (states, np.squeeze(meds), np.squeeze(rewards), next_states, np.squeeze(done_flags))


    def DTR(self):
        y_val = np.load('/Users/Downloads/unreal-master/model/y.npy')
        x_val = np.load('/Users/Downloads/unreal-master/model/val_x_3_base.npy')
        jac =[]
        qv = []

        BATCH_SIZE = self.config.batch_size
        GAMMA = self.config.gamma
        TAU = self.config.tau
        LRA = self.config.lra
        LRC = self.config.lrc
        epsilon = self.config.epsilon
        tiem_stamp = self.config.tiem_stamp
        lab_size = self.config.lab_size
        demo_size = self.config.demo_size
        max_reward = self.config.max_reward
        action_dim = self.config.med_size
        state_dim = self.config.state_dim
        np.random.seed(self.config.seed)
        episode_count = self.config.episode_count
        config_p = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        sess = tf.Session(config=config_p)
        K.set_session(sess)
        print('builda')

        actor = ActorNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRA, epsilon, tiem_stamp, med_size, lab_size, demo_size, di_size)
        print('buildc')
        critic = CriticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRC, epsilon, tiem_stamp, med_size, lab_size, demo_size, di_size, action_dim)

        print("Now we load the weight")
        try:
            actor.model.load_weights("actormodel.h5")
            critic.model.load_weights("criticmodel.h5")
            actor.target_model.load_weights("actormodel.h5")
            critic.target_model.load_weights("criticmodel.h5")
            print("Weight load successfully")
        except:
            print("Cannot find the weight")

        for i in range(episode_count):
                loss = 0
                # weight = df['prob']
                states, actions, rewards, new_states, dones, diseases, demos = self.batch(BATCH_SIZE)
                # states, actions, rewards, new_states, dones = self.process_batch(BATCH_SIZE,weight)
                len1 = states.shape[0]
                y_t = np.zeros(len1)
                ac = actor.target_model.predict([new_states, diseases, demos])
                target_q_values = critic.target_model.predict([new_states, ac, diseases,demos])
                target_q_values[target_q_values > max_reward] = max_reward
                target_q_values[target_q_values < -max_reward] = -max_reward

                for k in range((len1)):
                    if dones[k] == 1:
                        y_t[k] = rewards[k]
                    else:
                        y_t[k] = rewards[k] + GAMMA * target_q_values[k]

                lable = actions.copy()

                loss += critic.model.train_on_batch([states, actions, diseases, demos], y_t)
                a_for_grad = actor.model.predict([states, diseases, demos])
                grads = critic.gradients(states, diseases, demos, a_for_grad)
                actor.train(states, diseases, demos, lable, grads)


                actor.target_train()
                critic.target_train()

                if i % 10 == 0:
                    print(val_df_di.shape,val_sta.shape)
                    preds = actor.model.predict([val_df[lab_test].values, val_df_di, val_sta])
                    target_q_values = critic.target_model.predict([val_df[lab_test].values, preds, val_df_di, val_sta])
                    q = np.mean(target_q_values)
                    print('target_q_values',q)
                    # print("micro-auc", roc_auc_score(y_val[:,0], preds, average='micro'))
                    preds[preds >= 0.5] = 1
                    preds[preds < 0.5] = 0
                    j = jaccard_similarity_score(y_val, preds)
                    print('jaccard_similarity_score',j )
                    if i % 100 == 0:
                        np.save('ddpg_s_'+str(i)+'.npy', preds)
                        jac.append(j)
                        qv.append(q)
                    if i % 1000 == 0:
                        actor.model.save_weights("actormodel_s_"+str(i)+".h5", overwrite=True)
                        with open("actormodel_s_"+str(i)+".json", "w") as outfile:
                            json.dump(actor.model.to_json(), outfile)
                        critic.model.save_weights("criticmodel_s_"+str(i)+".h5", overwrite=True)
                        with open("criticmodel_s_"+str(i)+".json", "w") as outfile:
                            json.dump(critic.model.to_json(), outfile)
                        np.save('jac_s_'+str(i)+'.npy',np.array(jac))
                        np.save('qv_s_' + str(i)+'.npy', np.array(qv))




if __name__ == "__main__":
    model = SRL_RNN()
    model.DTR()



