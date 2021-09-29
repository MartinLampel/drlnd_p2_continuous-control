# Project 2: Continuous Control

* [Introduction](#introduction)
* [Learning Algorithm](#learning-algorithm)   
* [Project Structure](#project-structure)   
* [Implementation](#implementation)   
* [Results](#results)   
* [Future work](#ideas-for-future-work)

## Introduction

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

In order to solve the environment, our agent must achieve a score of +30 averaged across all 20 agents for 100 consecutive episodes.

To solve this task, I implemented the [DDPG algorithm](https://arxiv.org/pdf/1509.02971.pdf) algorithm. This algorithm is suitable to solve complex, high-dimensional tasks in the continuous domain. First, I cover some details of the algorithm for my understanding and give the reader background information. Then the implementation details and hyperparameters are explained. Finally, the results are presented. 



## Learning Algorithm

> DDPG combines the actor-critic approach with the insights from the Deep Q Network. DQN learn value functions 
> using deep neural networks in a stalbe and robust way. They utilize a replay buffer to minimize correlations between samples
> and a target Q network. DDPG is model free, off policy actor-critic algorithm that can learn high-dimensional, continuous action > spaces.  

## Project Structure

The code is written in PyTorch and Python3, executed in Jupyter Notebook

- Continuous_Control.ipynb	: Training and evaluation of the agent
- ddpg_agent.py	: An agent that implement the DDPG algorithm
- models.py	: DNN models for the actor and the critic
- replaybuffer.py : Implementation of experience replay buffer





## Implementation


### Results


## Ideas for Future Work