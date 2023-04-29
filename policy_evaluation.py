import numpy as np
import math
from KPI_nator import KPInator
import matplotlib.pyplot as plt

class PolicyEvaluationConfig:
    def __init__(self, env, debug, gamma, delta, iterations):
        self.env = env
        self.debug = debug
        self.gamma = gamma
        self.delta = delta
        self.iterations = iterations
        


class PolicyEvaluation:
    def __init__(self, conf: PolicyEvaluationConfig):
        """Calculates the value of being in a certain state"""
        self.env = conf.env
        # self.V = [np.zeros(self.env.observation_space.n) for _ in range(conf.iterations)]
        self.Vπ = np.zeros(self.env.observation_space.n)
        self.conf = conf

    def evaluate(self):
        """Evaluate the policy and adapt during the iterations"""
        for iteration in range(self.conf.iterations):
            Vπt = np.zeros(self.env.observation_space.n)

            # go through all the states at a certain time frame and return a value for that state
            for state in self.env.P:
                outer_sum = 0

                for action in self.env.P[state]:
                    # calculate the sum of next state values 
                    inner_sum  = 0
                    for probability, next_state, reward, _ in self.env.P[state][action]:
                        inner_sum += probability * (reward + self.conf.gamma * self.Vπ[next_state]) # calculate value for action in current state
                    
                    action_probability = 1 / (self.env.action_space.n)
                    outer_sum += action_probability * inner_sum

                Vπt[state] = outer_sum
                
            if self.conf.debug:
                analytics = KPInator()
                size = round(math.sqrt(self.env.observation_space.n))
                analytics.visualize_value_evaluation(Vπt, (size,size))
                plt.title("iteration: {}".format(iteration))

            if(np.max(np.abs(self.Vπ-Vπt)) < self.conf.delta):
                self.Vπ = Vπt   
                print("INFO: Converged at iteration {} ! (during training)".format(iteration))
                break;

            self.Vπ = Vπt

           

        