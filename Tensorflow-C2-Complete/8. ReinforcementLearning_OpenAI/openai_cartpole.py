# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 11:35:45 2020

@author: Lucky_Rathod
"""

'''
Now we will explore how to create an example environment with OpenAI Gym.
Goal of Cartpole Environment is to balance the pole on cart.
Actions allow us to move the cart left and right to attempt to balance pole.
Environment is numpy array with 4 floating point numbers .
-Horizontal Position
-Horizontal Velocity
-Angle of pole
-Angular Velocity.

You can grab these values from environment and use them for your agent

'''

import gym
print(gym.__version__)

# Make the environment, replace this string with any
# from the docs. (Some environments have dependencies)
env = gym.make('CartPole-v0')

# Reset the environment to default beginning
env.reset()

# Using _ as temp placeholder variable
# Render Environment for 1000 timesteps
# We are not using for var therfore _ is used
for _ in range(1000):
    # Render the env
    
    #Render will create pop up that allows you to view environment
    env.render()

    # Still a lot more explanation to come for this line!
    env.step(env.action_space.sample()) # take a random action
    
'''
Above Code will create a window in which cart will move from one location to another

'''

### OPEN AI GYM Observations

'''
step funtions returns usefull objects for our agent .

1. Observation

Env specific information representing environment observation.
Examples - Angles,Velocities,Game States

2. Reward

Amount of reward achieved by previous action.Agent will have to increase reward.

3. Done

Boolean variable indicating whether environment needs to be reset
Example - Game is Lost,Pole tipped over

4. Info 

Dictionary object with debugging information

'''    

import gym

# Make the environment, replace this string with any
# from the docs. (Some environments have dependencies)
env = gym.make('CartPole-v0')

print("Initial Observation")
# Reset the environment to default beginning
observation = env.reset()


for _ in range(2):
    
    #Will not render in these code
    action = env.action_space.sample()
    observation,reward,done,info = env.step(action) # take a random action
    
    print("Performed One Random Action")
    print('\n')
    print('observation')
    print(observation)
    print('\n')

    print('reward')
    print(reward)
    print('\n')

    print('done')
    print(done)
    print('\n')

    print('info')
    print(info)
    print('\n')

    
'''
These Gives an output of 4 Elements in array
It represents that poll standing straight up with the cart in middle position


#### OUTPUT ####

Initial Observation
Performed One Random Action


observation
[-0.02072113  0.1559783  -0.02845044 -0.34861167]


reward
1.0


done
False


info
{}

'''


### Open AI Gym Actions

'''
Now we will create a simple policy .Move the cart to right if pole falls to right 
and vice versa

'''   

import gym
env = gym.make('CartPole-v0')

## There are only Two Actions as you can see from print
print(env.action_space)
# #> Discrete(2)

## There are 4 observations 
print(env.observation_space)
# #> Box(4,)

#### Initial Observation    
observation = env.reset()

for t in range(1000):

    env.render()

    # 4 Observation
    cart_pos , cart_vel , pole_ang , ang_vel = observation

    # Move Cart Right if Pole is Falling to the Right

    # Angle is measured off straight vertical line
    if pole_ang > 0:
        # Move Right
        action = 1
    else:
        # Move Left
        action = 0

    # Perform Action
    observation , reward, done, info = env.step(action)
    
    
    
##############################################################################


           # NEURAL NETWORK
           
'''

Simple Neural Network that takes in observation array passes it through hidden layers
and outputs 2 probabilities,One for Left and another for right

Notice we dont just automatically choose the highest probability for our decision
This is to balance trying out new actions versus constantly choosing well known actions


Once we understand this network , we will explore how to take into account historic 
actions by learning about Policy Gradients
'''
           
           
##############################################################################

import tensorflow as tf
import gym
import numpy as np
###############################################
######## PART ONE: NETWORK VARIABLES #########
#############################################
 
# Observation Space has 4 inputs
num_inputs = 4

num_hidden = 4

# Outputs the probability it should go left
num_outputs = 1

initializer = tf.contrib.layers.variance_scaling_initializer()


###############################################
######## PART TWO: NETWORK LAYERS #########
#############################################

X = tf.placeholder(tf.float32, shape=[None,num_inputs])
hidden_layer_one = tf.layers.dense(X,num_hidden,activation=tf.nn.relu,kernel_initializer=initializer)
hidden_layer_two = tf.layers.dense(hidden_layer_one,num_hidden,activation=tf.nn.relu,kernel_initializer=initializer)

# Probability to go left
output_layer = tf.layers.dense(hidden_layer_two,num_outputs,activation=tf.nn.sigmoid,kernel_initializer=initializer)

# [ Prob to go left , Prob to go right]
probabilties = tf.concat(axis=1, values=[output_layer, 1 - output_layer])

# Sample 1 randomly based on probabilities
action = tf.multinomial(probabilties, num_samples=1)


init = tf.global_variables_initializer()



###############################################
######## PART THREE: SESSION #########
#############################################

saver = tf.train.Saver()

epi = 50
step_limit = 500
avg_steps = []
env = gym.make("CartPole-v1")
with tf.Session() as sess:
    init.run()
    for i_episode in range(epi):
        obs = env.reset()

        for step in range(step_limit):
            # env.render()
            action_val = action.eval(feed_dict={X: obs.reshape(1, num_inputs)})
            
            #action_val[0][0] - # 0 or 1
            obs, reward, done, info = env.step(action_val[0][0])
            if done:
                avg_steps.append(step)
                print('Done after {} steps'.format(step))
                break
print("After {} episodes the average cart steps before done was {}".format(epi,np.mean(avg_steps)))
env.close()


'''

OUR previous network didnt performed well . This maybe because we arent considering the history
of our actions , we are only consideringg a single previous action

'''

###############################################################################################



                # POLICY GRADIENT
                
                
###############################################################################################
                
                
import tensorflow as tf
import gym
import numpy as np





##########################
### VARIABLES ###########
########################

num_inputs = 4
num_hidden = 4
num_outputs = 1

learning_rate = 0.01

initializer = tf.contrib.layers.variance_scaling_initializer()


#################################
### CREATING THE NETWORK #######
###############################  

X = tf.placeholder(tf.float32, shape=[None, num_inputs])

hidden_layer = tf.layers.dense(X, num_hidden, activation=tf.nn.elu, kernel_initializer=initializer)
logits = tf.layers.dense(hidden_layer, num_outputs) 
outputs = tf.nn.sigmoid(logits)  # probability of action 0 (left)

probabilties = tf.concat(axis=1, values=[outputs, 1 - outputs])
action = tf.multinomial( probabilties, num_samples=1)

# Convert from Tensor to number for network training
y = 1. - tf.to_float(action)

########################################
### LOSS FUNCTION AND OPTIMIZATION ####
######################################
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
optimizer = tf.train.AdamOptimizer(learning_rate)

# https://stackoverflow.com/questions/41954198/optimizer-compute-gradients-how-the-gradients-are-calculated-programatically
# https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer


################################
#### GRADIENTS ################
##############################
gradients_and_variables = optimizer.compute_gradients(cross_entropy)



gradients = []
gradient_placeholders = []
grads_and_vars_feed = []

for gradient, variable in gradients_and_variables:
    gradients.append(gradient)
    gradient_placeholder = tf.placeholder(tf.float32, shape=gradient.get_shape())
    gradient_placeholders.append(gradient_placeholder)
    grads_and_vars_feed.append((gradient_placeholder, variable))


training_op = optimizer.apply_gradients(grads_and_vars_feed)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

######################################
#### REWARD FUNCTIONs ################
####################################
# CHECK OUT: https://medium.com/@awjuliani/super-simple-reinforcement-learning-tutorial-part-2-ded33892c724

def helper_discount_rewards(rewards, discount_rate):
    '''
    Takes in rewards and applies discount rate
    '''
    discounted_rewards = np.zeros(len(rewards))
    cumulative_rewards = 0
    for step in reversed(range(len(rewards))):
        cumulative_rewards = rewards[step] + cumulative_rewards * discount_rate
        discounted_rewards[step] = cumulative_rewards
    return discounted_rewards

def discount_and_normalize_rewards(all_rewards, discount_rate):
    '''
    Takes in all rewards, applies helper_discount function and then normalizes
    using mean and std.
    '''
    all_discounted_rewards = []
    for rewards in all_rewards:
        all_discounted_rewards.append(helper_discount_rewards(rewards,discount_rate))

    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [(discounted_rewards - reward_mean)/reward_std for discounted_rewards in all_discounted_rewards]

########################################
#### TRAINING SESSION #################
######################################

env = gym.make("CartPole-v0")

num_game_rounds = 10
max_game_steps = 1000
num_iterations = 650
discount_rate = 0.95

with tf.Session() as sess:
    sess.run(init)


    for iteration in range(num_iterations):
        print("Currently on Iteration: {} \n".format(iteration) )

        all_rewards = []
        all_gradients = []

        # Play n amount of game rounds
        for game in range(num_game_rounds):

            current_rewards = []
            current_gradients = []

            observations = env.reset()

            # Only allow n amount of steps in game
            for step in range(max_game_steps):

                # Get Actions and Gradients
                action_val, gradients_val = sess.run([action, gradients], feed_dict={X: observations.reshape(1, num_inputs)})

                # Perform Action
                observations, reward, done, info = env.step(action_val[0][0])

                # Get Current Rewards and Gradients
                current_rewards.append(reward)
                current_gradients.append(gradients_val)

                if done:
                    # Game Ended
                    break

            # Append to list of all rewards
            all_rewards.append(current_rewards)
            all_gradients.append(current_gradients)

        all_rewards = discount_and_normalize_rewards(all_rewards,discount_rate)
        feed_dict = {}


        for var_index, gradient_placeholder in enumerate(gradient_placeholders):
            mean_gradients = np.mean([reward * all_gradients[game_index][step][var_index]
                                      for game_index, rewards in enumerate(all_rewards)
                                          for step, reward in enumerate(rewards)], axis=0)
            feed_dict[gradient_placeholder] = mean_gradients

        sess.run(training_op, feed_dict=feed_dict)

    print('SAVING GRAPH AND SESSION')
    meta_graph_def = tf.train.export_meta_graph(filename='/new_saved_models/my-650-step-model.meta')
    saver.save(sess, '/new_saved_models/my-650-step-model')




#############################################
### RUN TRAINED MODEL ON ENVIRONMENT #######
###########################################

env = gym.make('CartPole-v0')

observations = env.reset()
with tf.Session() as sess:
    # https://www.tensorflow.org/api_guides/python/meta_graph
    new_saver = tf.train.import_meta_graph('/new_saved_models/my-650-step-model.meta')
    new_saver.restore(sess,'/new_saved_models/my-650-step-model')

    for x in range(500):
        env.render()
        action_val, gradients_val = sess.run([action, gradients], feed_dict={X: observations.reshape(1, num_inputs)})
        observations, reward, done, info = env.step(action_val[0][0])










