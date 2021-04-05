from itertools import product
import numpy as np
# import matplot.pyplot as plt
import numpy.random as rn
# conda install theano
import theano as th
import theano.tensor as T
import maxent
import csv
import collections
import pickle
from itertools import groupby
import pandas as pd
# import pymc3 as pm
# %matplotlib inline

# with open('raw_data2.csv','r') as file:
     # reader = csv.reader(file)
# print(type(df))
from itertools import accumulate
# from pd import Dataframe
df = pd.read_csv('raw_data2.csv')
# for subject_nr,df in df.groupby('subject_nr'):
    # print(df)
m = 1
list_new_variable3 = []
while m<len(df['xpos_get_response']):
      new_variable = df['xpos_get_response'][m].replace(',', '').replace('[','').replace(']','')
      new_variable2 = df['ypos_get_response'][m].replace(',', '').replace('[','').replace(']','')
      new_variable3 = df['subject_nr'][m]
      new_list = new_variable.split()
      len_new_list = len(new_list)
      new_list2 = new_variable2.split()
      last_index = len_new_list - 1
      if m != 1:
         first_list_new_variable = new_variable.split()
         first_list_new_variable2 = new_variable2.split()
         trajectories = [str(i)+','+str(j) for i,j in zip(first_list_new_variable, first_list_new_variable2)]
         trajectories2 = ['('+ str(i)+','+' '+str(j)+')' for i,j in zip(first_list_new_variable, first_list_new_variable2)]
         res = []
         denominator = len(trajectories2) - 1
         new_trajectories = []
         diff_trajectories = []
         for i in trajectories2: 
             if i not in res: 
                res.append(i)
         matrix = {
               k: {l: 0 for l in res}
               for k in res
         }
         for i in range(denominator): 
             # print(i)
             if trajectories2[i] == trajectories2[i+1]:
                matrix[trajectories2[i]][trajectories2[i+1]] += 1
             elif trajectories2[i] != trajectories2[i+1]:
                matrix[trajectories2[i]][trajectories2[i+1]] += 1
         a_file = open("data.pkl", "wb")
         pickle.dump(matrix, a_file)
         a_file.close()
         a_file = open("data.pkl", "rb")
         output = pickle.load(a_file)
         a_file.close()
         list_new_variable += new_variable.split()
         list_new_variable2 += new_variable2.split()
         list_new_variable3.append(new_variable3)
         len_new_list = len(list_new_variable)
      else: 
         list_new_variable = new_variable.split()
         list_new_variable2 = new_variable2.split()
         trajectories = [str(i)+','+' '+str(j) for i,j in zip(list_new_variable, list_new_variable2)]
         trajectories2 = ['('+ str(i)+','+' '+str(j)+')' for i,j in zip(list_new_variable, list_new_variable2)]
         new_trajectories = []
         for i in trajectories: 
             new_res = tuple(map(float, i.split(', '))) 
             new_trajectories.append(new_res)
         # print(new_trajectories)
         n_actions = len(new_trajectories)-1
         number_of_trajectories = 1
         length_of_trajectories = len(new_trajectories)
         # print(new_trajectories)
         trajectories1 = np.array(new_trajectories)
         trajectories1 = trajectories1.astype(int)
         # print(trajectories1)
         new_trajectories2 = trajectories1.reshape(number_of_trajectories,length_of_trajectories,2)
         # print(new_trajectories2)
         # print(new_trajectories2.shape)
         # print(new_trajectories2[0, 0, 0])
         new_counter = collections.Counter(new_trajectories)
         new_df = pd.DataFrame.from_dict(new_counter, orient='index').reset_index()
         # print(new_df)
         new_df_count = new_df['index'].count()
         # print(new_df_count)
         res = []
         for i in trajectories: 
             if i not in res: 
                res.append(i)
         res2 = []
         denominator2 = len(trajectories2) - 1
         for i in trajectories2: 
             if i not in res2: 
                res2.append(i)
         convert_res = []
         for i in res:
             new_res = tuple(map(float, i.split(', '))) 
             convert_res.append(new_res)
         convert_res_len = len(convert_res)
         x_coordinates = [p[0] for p in convert_res]
         y_coordinates = [p[1] for p in convert_res]
         first_new_x_y_coordinates = [str(i-.5)+','+' '+str(j-.5) for i,j in zip(x_coordinates, y_coordinates)]
         first_new_x_y_coordinates_converted = []
         for i in first_new_x_y_coordinates:
             new_res = tuple(map(float, i.split(', '))) 
             first_new_x_y_coordinates_converted.append(new_res)
         first_new_x_coordinates = [p[0] for p in first_new_x_y_coordinates_converted]
         first_new_y_coordinates = [p[1] for p in first_new_x_y_coordinates_converted]
         added_first_x_y_coordinates = []
         for i in range(len(first_new_x_coordinates)):
             added_first_x_y_coordinates.append(first_new_x_coordinates[i] + first_new_y_coordinates[i])
         second_new_x_y_coordinates = [str(i-.5)+','+' '+str(j+.5) for i,j in zip(x_coordinates, y_coordinates)]
         second_new_x_y_coordinates_converted = []
         for i in second_new_x_y_coordinates:
             new_res = tuple(map(float, i.split(', '))) 
             second_new_x_y_coordinates_converted.append(new_res)
         second_new_x_coordinates = [p[0] for p in second_new_x_y_coordinates_converted]
         second_new_y_coordinates = [p[1] for p in second_new_x_y_coordinates_converted]
         added_second_x_y_coordinates = []
         for i in range(len(second_new_x_coordinates)):
             added_second_x_y_coordinates.append(second_new_x_coordinates[i] + second_new_y_coordinates[i])
         third_new_x_y_coordinates = [str(i+.5)+','+' '+str(j+.5) for i,j in zip(x_coordinates, y_coordinates)]
         third_new_x_y_coordinates_converted = []
         for i in third_new_x_y_coordinates:
             new_res = tuple(map(float, i.split(', '))) 
             third_new_x_y_coordinates_converted.append(new_res)
         third_new_x_coordinates = [p[0] for p in third_new_x_y_coordinates_converted]
         third_new_y_coordinates = [p[1] for p in third_new_x_y_coordinates_converted]
         added_third_x_y_coordinates = []
         for i in range(len(third_new_x_coordinates)):
             added_third_x_y_coordinates.append(third_new_x_coordinates[i] + third_new_y_coordinates[i])
         fourth_new_x_y_coordinates = [str(i+.5)+','+' '+str(j-.5) for i,j in zip(x_coordinates, y_coordinates)]
         fourth_new_x_y_coordinates_converted = []
         for i in fourth_new_x_y_coordinates:
             new_res = tuple(map(float, i.split(', '))) 
             fourth_new_x_y_coordinates_converted.append(new_res)
         fourth_new_x_coordinates = [p[0] for p in fourth_new_x_y_coordinates_converted]
         fourth_new_y_coordinates = [p[1] for p in fourth_new_x_y_coordinates_converted]
         added_fourth_x_y_coordinates = []
         for i in range(len(fourth_new_x_coordinates)):
             added_fourth_x_y_coordinates.append(fourth_new_x_coordinates[i] + fourth_new_y_coordinates[i])
         new_df['added_first_new_x_y_coordinates'] = added_first_x_y_coordinates
         new_df['added_second_new_x_y_coordinates'] = added_second_x_y_coordinates
         new_df['added_third_new_x_y_coordinates'] = added_third_x_y_coordinates
         new_df['added_fourth_new_x_y_coordinates'] = added_fourth_x_y_coordinates
         # new_df.columns = ['cell_id','number_of_visits', 'first_new_x_y_coordinates', 'second_new_x_y_coordinates', 'third_new_x_y_coordinates']
         new_df = new_df.iloc[: , 1:]
         new_df.columns = ['number_of_visits', 'added_first_new_x_y_coordinates', 'added_second_new_x_y_coordinates', 'added_third_new_x_y_coordinates', 'added_fourth_new_x_y_coordinates']
         # print(new_df)
         numOfRows = new_df.shape[0]
         new_matrix =  np.array(new_df)
         feature_matrix = new_matrix.reshape(numOfRows,5)
         # print(feature_matrix)
         matrix = {
               k: {l: 0 for l in res2}
               for k in res2
         }
         a_file = open("data.pkl", "wb")
         pickle.dump(matrix, a_file)
         a_file.close()
         a_file = open("data.pkl", "rb")
         output = pickle.load(a_file)
         a_file.close()
         new_trajectories = []
         diff_trajectories = []
         for i in range(denominator2): 
             if trajectories2[i] == trajectories2[i+1]:
                matrix[trajectories2[i]][trajectories2[i+1]] += (1/n_actions)
             elif trajectories2[i] != trajectories2[i+1]:
                matrix[trajectories2[i]][trajectories2[i+1]] += (1/n_actions)
         for key in matrix.keys():
             matrix[key] = list(matrix[key].values())
         new_array = []
         for value in matrix.values():
             new_value = [value] * n_actions
             new_array.append(new_value)
             # print(type(value))
             # print(value)
         new_array = np.array(new_array)
         transition_probability = new_array.reshape(convert_res_len,n_actions,convert_res_len)
         FLOAT = th.config.floatX
         def find_svf(n_states, trajectories):
             svf = np.zeros(n_states)
             for trajectory in trajectories:
                 for state, _, _ in trajectory:
                      # print(state)
                      # print(_)
                      # print(_)
                      svf[state] += 1
             svf /= trajectories.shape[0]
             return th.shared(svf, "svf", borrow=True)
         def optimal_value(n_states, n_actions, transition_probabilities,reward,discount,threshold=1e-2):      
             v = T.zeros(n_states, dtype=FLOAT)
             def update(s, prev_diff, v, reward, tps):
                 max_v = float("-inf")
                 v_template = T.zeros_like(v)
                 for a in range(n_actions):
                     tp = tps[s, a, :]
                     max_v = T.largest(max_v, T.dot(tp, reward + discount*v))
                 new_diff = abs(v[s] - max_v)
                 if T.lt(prev_diff, new_diff):
                    diff = new_diff
                 else:
                    diff = prev_diff
                 return (diff, T.set_subtensor(v_template[s], max_v)), {}
             def until_converged(diff, v):
                 (diff, vs), _ = th.scan(
                         fn=update,
                         outputs_info=[{"initial": diff, "taps": [-1]},
                                       None],
                         sequences=[T.arange(n_states)],
                         non_sequences=[v, reward, transition_probabilities])
                 return ((diff[-1], vs.sum(axis=0)), {},
                         th.scan_module.until(diff[-1] < threshold))

             (_, vs), _ = th.scan(fn = until_converged,
                                  outputs_info=[
                                     {"initial": getattr(np, FLOAT)(float("inf")),
                                      "taps": [-1]},
                                     {"initial": v,
                                      "taps": [-1]}],
                                  n_steps=1000)

             return vs[-1]
         def find_policy(n_states,n_actions,transition_probabilities,reward, 
                          discount,threshold=1e-2,v=None):
              if v is None:
                 v = optimal_value(n_states, n_actions, transition_probabilities, reward,
                              discount, threshold)
              Q = T.zeros((n_states, n_actions))
              def make_Q(i, j, tps, Q, reward, v):
                  Q_template = T.zeros_like(Q)
                  tp = transition_probabilities[i, j, :]
                  return T.set_subtensor(Q_template[i, j], tp.dot(reward + discount*v)),{}
              prod = np.array(list(product(range(n_states), range(n_actions))))
              state_range = th.shared(prod[:, 0])
              action_range = th.shared(prod[:, 1])
              Qs, _ = th.scan(fn=make_Q,
                              outputs_info=None,
                              sequences=[state_range, action_range],
                              non_sequences=[transition_probabilities, Q, reward, v])
              Q = Qs.sum(axis=0)
              Q -= Q.max(axis=1).reshape((n_states, 1))  # For numerical stability.
              Q = T.exp(Q)/T.exp(Q).sum(axis=1).reshape((n_states, 1))
              return Q
         def find_expected_svf(n_states, r, n_actions, discount,
                               transition_probability, trajectories):  
             n_trajectories = trajectories.shape[0]
             trajectory_length = trajectories.shape[1]
             policy = find_policy(n_states, n_actions, transition_probability, r, discount)
             start_state_count = T.extra_ops.bincount(trajectories[:, 0, 0],
                                             minlength=n_states)
             p_start_state = start_state_count.astype(FLOAT)/n_trajectories
             def state_visitation_step(i, j, prev_svf, policy, tps):
                 svf = prev_svf[i] * policy[i, j] * tps[i, j, :]
                 return svf, {}
             prod = np.array(list(product(range(n_states), range(n_actions))))
             state_range = th.shared(prod[:, 0])
             action_range = th.shared(prod[:, 1])
             def state_visitation_row(prev_svf, policy, tps, state_range, action_range):
                 svf_t, _ = th.scan(fn=state_visitation_step,
                           sequences=[state_range, action_range],
                           non_sequences=[prev_svf, policy, tps])
                 svf_t = svf_t.sum(axis=0)
                 return svf_t, {}
             svf, _ = th.scan(fn=state_visitation_row,
                          outputs_info=[{"initial": p_start_state, "taps": [-1]}],
                          n_steps=trajectories.shape[1]-1,
                          non_sequences=[policy, transition_probability, state_range,
                          action_range])
             return svf.sum(axis=0) + p_start_state
         def irl(structure, feature_matrix, n_actions, discount, transition_probability,
             trajectories, epochs, learning_rate, initialisation="normal", l1=0.1,
             l2=0.1):
             n_states, d_states = feature_matrix.shape
             # print(feature_matrix)
             n_states = int(n_states)
             # print(n_states)
             transition_probability = th.shared(transition_probability, borrow=True)
             trajectories = th.shared(trajectories, borrow=True)
             n_layers = len(structure)-1
             weights = []
             hist_w_grads = []  # For AdaGrad.
             biases = []
             hist_b_grads = []  # For AdaGrad.
             for i in range(n_layers):
                 shape = (structure[i+1], structure[i])
                 if initialisation == "normal":
                     matrix = th.shared(rn.normal(size=shape), name="W", borrow=True)
                 else:
                     matrix = th.shared(rn.uniform(size=shape), name="W", borrow=True)
                 weights.append(matrix)
                 hist_w_grads.append(th.shared(np.zeros(shape), name="hdW", borrow=True))
                 shape = (structure[i+1], 1)
                 if initialisation == "normal":
                     matrix = th.shared(rn.normal(size=shape), name="b", borrow=True)
                 else:
                     matrix = th.shared(rn.uniform(size=shape), name="b", borrow=True)
                 biases.append(matrix)
                 hist_b_grads.append(th.shared(np.zeros(shape), name="hdb", borrow=True))
             if initialisation == "normal":
                 α = th.shared(rn.normal(size=(1, structure[-1])), name="alpha",
                            borrow=True)
             else:
                 α = th.shared(rn.uniform(size=(1, structure[-1])), name="alpha",
                            borrow=True)
             hist_α_grad = T.zeros(α.shape)  # For AdaGrad.
             adagrad_epsilon = 1e-6  # AdaGrad numerical stability.
             s_feature_matrix = T.matrix("x")
             print(th.tensor.shape(s_feature_matrix))
             # rows = len(s_feature_matrix)
             # columns = len(s_feature_matrix[0])
             # print(rows)
             # print(columns)
             # print(np.shape(s_feature_matrix))
             # s_feature_matrix = T.matrix("x").astype(float)
             # print(s_feature_matrix)
             φs = [s_feature_matrix.T]
             # print(φs
             for W, b in zip(weights, biases):
                 φ = T.nnet.sigmoid(th.compile.ops.Rebroadcast((0, False), (1, True))(b) + W.dot(φs[-1]))
                 φs.append(φ)
             # print(φs[-1])
             # r = α.dot(φs[-1]).reshape((n_states,))
             r = α.dot(φs[-1])
             # print(r)
             # r = r.reshape((n_states,))
             r = r.reshape((2,))
             r = (r - r.mean())/r.std()
             expected_svf = find_expected_svf(n_states, r,
                                           n_actions, discount,
                                           transition_probability,
                                           trajectories)
             number_column = new_df.loc[:,'number_of_visits']
             numbers = number_column.values
             numbers = np.floor(numbers)
             # print(numbers)
             svf = maxent.find_svf(n_states, numbers,new_trajectories2)
             updates = []
             α_grad = φs[-1].dot(svf - expected_svf).T
             hist_α_grad += α_grad**2
             adj_α_grad = α_grad/(adagrad_epsilon + T.sqrt(hist_α_grad))
             updates.append((α, α + adj_α_grad*learning_rate))
             def grad_for_state(s, theta, svf_diff, r):
                 regularisation = abs(theta).sum()*l1 + (theta**2).sum()*l2
                 return svf_diff[s] * T.grad(r[s], theta,) - regularisation, {}
             for i, W in enumerate(weights):
                 w_grads, _ = th.scan(fn=grad_for_state,
                                   sequences=[T.arange(n_states)],
                                   non_sequences=[W, svf - expected_svf, r])
                 w_grad = w_grads.sum(axis=0)
                 hist_w_grads[i] += w_grad**2
                 adj_w_grad = w_grad/(adagrad_epsilon + T.sqrt(hist_w_grads[i]))
                 updates.append((W, W + adj_w_grad*learning_rate))
             for i, b in enumerate(biases):
                 b_grads, _ = th.scan(fn=grad_for_state,
                             sequences=[T.arange(n_states)],
                             non_sequences=[b, svf - expected_svf, r])
                 b_grad = b_grads.sum(axis=0)
                 hist_b_grads[i] += b_grad**2
                 adj_b_grad = b_grad/(adagrad_epsilon + T.sqrt(hist_b_grads[i]))
                 updates.append((b, b + adj_b_grad*learning_rate))
             # print(feature_matrix[...,1].ravel())
             # matrix_number_of_visits = np.squeeze(feature_matrix[...,1].ravel())
             # print(matrix_number_of_visits)
             # matrix_number_of_visits = np.floor(matrix_number_of_visits)
             matrix_number_of_visits = feature_matrix.astype(float)
             matrix_number_of_visits = np.matrix(matrix_number_of_visits)
             # print(matrix_number_of_visits.shape)
             reshaped_to_2d = np.array(matrix_number_of_visits)
             # print(reshaped_to_2d)
             # print(reshaped_to_2d)
             reshaped_to_2d_reshape = reshaped_to_2d.reshape(n_states,5)
             # print(reshaped_to_2d_reshape.shape[0])
             # print(reshaped_to_2d_reshape)
             # reshaped_to_2d_reshape = reshaped_to_2d_reshape.reshape(reshaped_to_2d_reshape.shape[0],1)
             # print(type(reshaped_to_2d_reshape))
             # print(reshaped_to_2d_reshape)
             # print(reshaped_to_2d_reshape.shape)
             # print(matrix_number_of_visits)
             # print(type(matrix_number_of_visits))
             # reshaped_to_2d = np.reshape(matrix_number_of_visits, (-1, 2))
             # print(type(reshaped_to_2d))
             # print(reshaped_to_2d.shape)
             # reshaped_to_2d_reshape =  reshaped_to_2d.reshape(10,3)
             # print(reshaped_to_2d_reshape)
             # print(s_feature_matrix)
             # print(updates)
             # print(r)
             # print(s_feature_matrix.shape)
             # print(s_feature_matrix[0])
             # train = th.function([s_feature_matrix], updates=updates)
             print([s_feature_matrix])
             # print(dim([s_feature_matrix]))
             train = th.function([s_feature_matrix], updates=updates, outputs=r)
             print(updates)
             # train = th.function([s_feature_matrix], outputs=r)
             # print(train)
             run = th.function([s_feature_matrix], outputs=r)
             # print(reshaped_to_2d.shape)
             # print(reshaped_to_2d)
             # first_two_columns = reshaped_to_2d_reshape[:,:2]
             # next_three_colums = reshaped_to_2d_reshape[:, [2, 4]]
             # print(first_two_columns)
             # print(next_three_colums)
             # print(reshaped_to_2d_reshape[1])
             for e in range(epochs):
                 # print(e)
                 # print(train)
                 # reward = train(reshaped_to_2d_reshape)
                 reward = train(reshaped_to_2d_reshape[0:2, :])
                 # reward = train(first_two_columns[0],next_three_colums[0])
                 # print(reward)
             # return reward.reshape((n_states,))
             return reward.reshape((2,))
      structure = (5,2,1)
      epochs = 1
      learning_rate = 0.5
      discount = 0.5
      print(irl(structure,feature_matrix,n_actions,discount,
                 transition_probability,new_trajectories2,epochs,
                 learning_rate,initialisation="normal",l1=0.1,l2=0.1))              
      m += 2
 # for key, value in matrix.items(): 
             # print(key,value)
         # for value in matrix.items(): 
              # value.replace('(   ,  ): ', '')
         # print(matrix)
         # data = list(matrix.items())
         # print(data)
         # an_array = np.array(data)
         # new_array = an_array.reshape(numOfRows,n_actions,numOfRows)
         # print(an_array)
         # print(an_array.shape)
      # print(matrix)
          
       # print(matrix)
       # new_list = [eval(i) for i in res]
       # df_t = pd.DataFrame(new_list).T
       # df_t = pd.DataFrame(new_list)
       # print(df_t)
       # print(type(df_t))
       # print(new_list)
       # matrix = np.array(res)
       # print(matrix)
       # print(matrix)
       # mat = [res,res]
       # mat2 = map(list, zip(*mat))
       # mat = np.array([res,res]).T
       # print(mat) 
       # matrix_values = np.array([res,res])
       # print(matrix_values)
       # print(mat2)
       # matrix1 = [p for p in zip(res,res)]
       # print(matrix1)
       # Res_List = [res,res]
       # print(np.array(Res_List))
       # print(res)
       # size = 2
       # new_res = [res for _ in range(size)]    # _ is throwaway variable name
       # shape = (len(res),len(res))
       # print(new_res) 
       # print(type(trajectories))
       # print(res)
       # print(len(res))
       # res_index = len(res)-1
       # for i in range(res_index):
       # denominator = len(trajectories) - 1
       # print(trajectories)
       # print(res)
       # new_res = []
       # for i in range(denominator): 
           # res = tuple(map(int, trajectories[i].split(','))) 
           # new_res.append(res)
           # print(res)
       # for i in new_res:
       # print(new_res)
       # print(type(res))
       # print(res)
       # new_denominator = len(new_res) - 1
         # for i in range(denominator): 
             # if trajectories2[i] == trajectories2[i+1]:
                # matrix[trajectories2[i]][trajectories2[i+1]] += 1
                # new_trajectories.append(trajectories2[i])
                # new_trajectories.append(str(trajectories[i]) + '->' + str(trajectories[i+1]))
             # else:
                # diff_trajectories.append([trajectories2[i]] [trajectories2[i+1]])
                # matrix[trajectories2[i]][trajectories2[i+1]] += 1
                # diff_trajectories.append(str(trajectories2[i]) +'->' +str(trajectories2[i+1]))
       # print(diff_trajectories)
       # for i in range(new_denominator): 
           # if new_res[i] == new_res[i+1]:
              # new_trajectories.append(new_res[i])
              # new_trajectories.append(str(trajectories[i]) + '->' + str(trajectories[i+1]))
           # else:
              # diff_trajectories.append(str(new_res[i]) +'->' +str(new_res[i+1]))
       # print(new_trajectories)
       # print(matrix)
       # print(matrix.keys())
       # for m in (matrix.keys()): 
           # print(m)
           # print(type(m))
           # if m in new_trajectories:
              # matrix[m][m] += 1
           # new_m = ''.join(m)
           # print(m)
       # print(new_trajectories)
       # print(len(new_trajectories))
       # print(diff_trajectories)
       # print(len(diff_trajectories))
       # print(len(new_trajectories))
       # new_list = [x[0] for x in groupby(trajectories)]
       # new_list2 = [(x,) for x in new_list]
       # print(new_list)
       # print(len(new_list))
       # res_new_list = [] 
       # print(res_new_list)
       # print(new_list2)
       # print(denominator)
       # my_final_list = set(trajectories)
       # print(list(my_final_list))
       # list_new_variable3.append(new_variable3)
       # print(list_new_variable3)
       # len_new_list = len(list_new_variable) 
# print(list_new_variable)
# print(list_new_variable2)
# print(list_new_variable3)
trajectories = [str(k)+' '+'('+ str(i)+','+str(j)+')' for i,j,k in zip(list_new_variable, list_new_variable2,list_new_variable3)]
trajectories2 = ['('+ str(i)+','+str(j)+')' for i,j in zip(list_new_variable, list_new_variable2)]
# print(len(trajectories))
# print(type(trajectories))
# print(trajectories)
trajectories_df = pd.DataFrame(trajectories)
trajectories_df.columns = ["subject_nr_and_x_y_coordinates"]
# df[['Name', 'Lastname']] = df.name.str.split(' ', expand = True)
trajectories_df[['subject_nr', 'x_y_coordinates']] = trajectories_df.subject_nr_and_x_y_coordinates.str.split(' ',expand=True)
# print(trajectories_df)
new_trajectories_df = trajectories_df.drop(columns=['subject_nr_and_x_y_coordinates'])
# new_trajectories_df = new_trajectories_df.dropna()
# print(new_trajectories_df)
# print(len(new_trajectories_df))
# print(len(new_trajectories_df))
# print(new_trajectories_df.shape[0])
results1 = list(map(int, list_new_variable))
# print(len(results1))
results2 = list(map(int, list_new_variable2))
# print(len(results2))
new_trajectories_df_second_column = ['('+str(i)+','+str(j-1)+')' for i,j in zip(results1, results2)]
new_trajectories_df_second_column_second = new_trajectories_df_second_column[0:1139]
# print(len(new_trajectories_second_column))
new_trajectories_df_second_column = pd.DataFrame(new_trajectories_df_second_column)
new_trajectories_df_second_column.columns = ["First_new_x_and_y_coordinates"]
new_trajectories_df = pd.concat([new_trajectories_df,new_trajectories_df_second_column], axis=1)
# new_trajectories_df = new_trajectories_df.head(1140)
# print(new_trajectories_df)
# trajectories_df.columns = ["subject_nr_and_x_y_coordinates"]
new_trajectories_df_third_column = ['('+str(i-.5)+','+str(j+.5)+')' for i,j in zip(results1, results2)]
# print(new_trajectories_df_third_column)
new_trajectories_df_third_column_second = new_trajectories_df_third_column[0:1139]
# print(new_trajectories_df_third_column_second)
# print(len(new_trajectories_df_third_column))
new_trajectories_df_third_column  = pd.DataFrame(new_trajectories_df_third_column)
new_trajectories_df_third_column.columns = ["Second_new_x_and_y_coordinates"]
new_trajectories_df = pd.concat([new_trajectories_df,new_trajectories_df_third_column], axis=1)
new_trajectories_df_fourth_column = ['('+str(i+.5)+','+str(j+.5)+')' for i,j in zip(results1, results2)]
new_trajectories_df_fourth_column_second = new_trajectories_df_fourth_column[0:1139]
# print(len(new_trajectories_df_fourth_column))
new_trajectories_df_fourth_column  = pd.DataFrame(new_trajectories_df_fourth_column)
new_trajectories_df_fourth_column.columns = ["Third_new_x_and_y_coordinates"]
new_trajectories_df = pd.concat([new_trajectories_df,new_trajectories_df_fourth_column], axis=1)
new_trajectories_df = new_trajectories_df.dropna()
new_x_values = []
new_y_values = []
for i in range(len(new_trajectories_df_second_column_second)):
    ans = trajectories2[i].split(',', 1)[0].replace('(','')
    # print(ans)
    ans = float(ans)
    ans1 = new_trajectories_df_second_column_second[i].split(',', 1)[0].replace('(','')# maxsplit = 1; 
    ans1 = float(ans1)
    ans2 = new_trajectories_df_third_column_second[i].split(',', 1)[0].replace('(','')
    # print(ans2)
    ans2 = float(ans2)
    # print(ans2)
    ans3 = new_trajectories_df_fourth_column_second[i].split(',', 1)[0].replace('(','')
    ans3 = float(ans3)
    new_ans_x_value = (ans+ans1+ans2+ans3)/4
    new_x_values.append(new_ans_x_value)
    ans_y_value = trajectories2[i].split(',', 1)[1].replace(')','')
    ans_y_value = float(ans_y_value)
    ans1_y_value = new_trajectories_df_second_column_second[i].split(',', 1)[1].replace(')','')
    ans1_y_value = float(ans1_y_value)
    ans2_y_value = new_trajectories_df_third_column_second[i].split(',', 1)[1].replace(')','')
    ans2_y_value = float(ans2_y_value)
    ans3_y_value = new_trajectories_df_fourth_column_second[i].split(',', 1)[1].replace(')','')
    ans3_y_value = float(ans3_y_value)
    new_ans_y_value = (ans_y_value+ ans1_y_value + ans2_y_value + ans3_y_value)/4
    new_y_values.append(new_ans_y_value)
    # print(ans) 
# print(new_x_values)
# print(new_y_values)
cell_id = ['('+ str(i)+','+str(j)+')' for i,j in zip(new_x_values, new_y_values)]
# print(type(cell_id))
unique_cell_id = []
for i in cell_id: 
    if i not in unique_cell_id: 
       unique_cell_id.append(i)
# print(len(unique_cell_id))
cell_id = pd.DataFrame(cell_id)
cell_id.columns = ["cell_id"]
new_trajectories_df = pd.concat([cell_id,new_trajectories_df], axis=1)
# new_trajectories_df['cell_id_occurences'] = new_trajectories_df['cell_id'].value_counts()
new_value_counts = new_trajectories_df['cell_id'].value_counts()
# print(type(new_value_counts))
# new_value_counts = new_trajectories_df.value_counts()
# print(new_value_counts)
# print(new_value_counts['(17.0,421.0)'])
dataTypeSeries = new_trajectories_df.dtypes
# print('Data type of each column of Dataframe :')
# print(dataTypeSeries)
new_trajectories_df['cell_id_values'] = ""
# df = pd.DataFrame([{'c1':10, 'c2':100}, {'c1':11,'c2':110}, {'c1':12,'c2':120}])
# print(df)
# for index, row in df.iterrows():
    # print(index)
    # print(row['c1'], row['c2'])
for index,row in new_trajectories_df.iterrows():
    row['cell_id_values'] = new_value_counts[str(row.cell_id)]
    # print(new_trajectories_df['cell_id_values'])
    # print(row.cell_id)
# print(new_trajectories_df)
# new_value_counts_df = pd.DataFrame(new_value_counts)
# print(new_value_counts_df)
# index = new_value_counts_df.index
# number_of_rows = len(index)-1
# print(number_of_rows)
# for index, row in new_value_counts_df.iterrows():
     # print(index, row)
# for i in range(number_of_rows):
    # print(new_value_counts_df[i])
# print(new_trajectories_df['cell_id'].count())
# new_trajectories_df['cell_id_values'] = new_value_counts[new_trajectories_df['cell_id']
# print(new_trajectories_df['cell_id'].value_counts())
# print(type(new_trajectories_df['cell_id'].value_counts()))
# print(new_trajectories_df)
# def foo(x_y_coordinates):
    # return pd.Series([lat + 10, lon * 100])
# trajectories_df[""]= trajectories_df[""].str.split(" ", n = 1, expand = True) 
# print(trajectories_df)
# trajectories_df[['subject_nr', 'x_y_coordinates']] = trajectories_df.subject_nr_and_x_y_coordinates.split(" ",expand=True)
# print(len(trajectories_df.columns))
# print(trajectories_df)
# for subject_nr,new_trajectories_df in new_trajectories_df.groupby('subject_nr'):
    # print(new_trajectories_df)
# print(new_trajectories_df.groupby('cell_id').count()) 
# print(new_trajectories_df)
# print(type(new_trajectories_df))
# for cell_id,new_trajectories_df in new_trajectories_df.groupby('cell_id'):
    # print(new_trajectories_df)
# Newtrajectories= [[x] for x in trajectories]
# print(trajectories)
# print(Output_trajectories_array) 