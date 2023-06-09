#=================== imports ===================#
import gym
from policy_evaluation import PolicyEvaluation, PolicyEvaluationConfig

#==================== setup ====================#
DEBUG = False #if you want the debug results for in between training, you can set DEBUG = True
size = 4

map_name = "8x8" if size == 8  else "4x4"
env = gym.make('FrozenLake-v1', render_mode="human", map_name=map_name, is_slippery=False)
env.reset()
env.render()

#==================== debugging and testing before creating anything ====================#
print("DEBUG: observation_space: {}".format(env.observation_space.n)) # 4 x 4 board = 16 observations possible
print("DEBUG: action_space: {}".format(env.action_space.n)) # actions: left 0, down 1, right 2, up 3 = 4 possible actions
# we can ask the environment at a particular state, when you want to perform action x, what the probabilities are
# using env.P[state][action]
# env.P[state, action] returns format (transition probability, next state, reward, is terminal state)
print("DEBUG: determening probability for state {}, action {}: {}".format(0, 0, env.P[0][0]))

#==================== hyper parameters ====================#
gamma = 0.9
delta = 10**(-6)
iterations=100 # converges at 32 for 4x4 and 47 for 8x8
config = PolicyEvaluationConfig(env, DEBUG, gamma, delta, iterations)

print("=======================================================")
print("INFO: Value function takes future into account with (gamma) {}".format(gamma))
print("INFO: value function converges if the difference between the new value state and the old value state is less then {}".format(delta))
print("INFO: The algorithm tries to converge within the {} itterations".format(iterations))
print("=======================================================")

#==================== Create policy value per state ====================# 
print("INFO: Training.......")
policy_evaluation = PolicyEvaluation(config)
policy_evaluation.evaluate()

#==================== play ====================#
print("INFO: Playing the game.......")
state = 0
last_state_value = 0
best_next_state = 0
step_action = -1
terminated = False

while(terminated == False):
    # we are in a current state
    # choose the next_state with the highest value in the value table.
    for action in env.P[state]:
        for probability, next_state, reward, _ in env.P[state][action]:
            if policy_evaluation.Vπ[next_state] > last_state_value: 
                last_state_value = policy_evaluation.Vπ[next_state]
                best_next_state = next_state
               

    # map next state to action
    if state - 1 == best_next_state : step_action = 0 # left
    if state + 1 == best_next_state : step_action = 2 # right
    if state - size == best_next_state : step_action = 3 # up
    if state + size == best_next_state : step_action = 1 # down

    #take action
    observation, reward, terminated, truncated, info = env.step(step_action)
    state = observation

    #render and reset
    env.render()

print("INFO: Ended the game.......")
env.close()

