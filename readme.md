# Gym (Frozen Lake) solution using Policy Evaluation
In reinforcement learning, There are a lot of techniques that you can follow.
Q-learning is one of them and is used the most at beginner level projects. Most examples of frozen lake are using a Q table to train there agent.

In this excercise, I am trying to create an easier solution by calculating the value for each state. This is done by using the `bellman equation` to calculate the value of a state towards the next steps that can be taken. In this example, we are trying to create a `value function` to get the optimal `Vπ (value policy)`. The next chapters contain information about prerequisites, how to setup your project, the approach taken, the results, and further suggestions.

## Prerequisites
Here are some things to consider installing before you can use this project properly.
### Tools
- Python
- Git
- Visual studio code
### Packages
- seaborn
- matplotlib
- gym
- math
- numpy

## Setup Project
> :warning:
> If you want to run this application in debug mode, you can change the `DEBUG` flag in `main.py` to be `True`

you can use the project by opening a command prompt and using the command:
```bash
git clone git@github.com:ericvdbergeFontys/AI-Frozen-lake.git
```
you can now start the application by typing:
```bash
python main.py
```

## Approach
Nice! you have set up your environment, so you can work with the solution! But how did we come to the current solution?<br>

There are a lot of models / techniques that you can use to create a reinforcement learning agent. Nowadays, we can't just take the one that feels the best. We have a lot of aspects to take into account, like the environment crisis, and limited compute power. My `goal of the project` is to make a solution that is `not over-engineered`, and is as simple as possible to complete frozen lake while using the `bellman equation`.

After doing some research, I found the following [blog](https://aleksandarhaber.com/iterative-policy-evaluation-algorithm-in-python-reinforcement-learning-tutorial/) that explained the theory of `the Iterative Policy Evaluation Algorithm`. This uses a value function that would calculate a value of a state towards the goal. So the further away from the goal, the smaller the value, the closer it is to the goal, the bigger the value.<br>

That is where the blog ended, but not where I ended. So from that point, we have a map of values for every state. But how do we determine which action to take? That is where i made a loop where it plays the game as follows:<br>
1. Select a state (0 at start)
2. Check which actions are possible (right and down in state 0)
3. **Check which action will bring you to the state with the highest value in the value table**
4. Map the best next state to the appropiate action
5. Make a move
6. Set the new state based on where you ended up (and repeat, go to step 1 again)

This loops breaks when the game itself terminates, which is when we get to the goal.

## Results
In the first iteration of creating the value table, we are going to check if there is a state that has a high value (because it leeds to a win). As you can see in the figure below, it will add a 0.25 value to the state before we get to the goal. 
<img src="./images/value policy map - iteration 1.png"/>
This means that it is really valuable to be in the state (V14), because we can then win the game. Lets calculate what happens in the second iteration.
<img src="./images/value policy map - iteration 2.png"/>
In the second generation, we now calculate that it is valuable to be in the spot next to the state that makes us win. The `bellman equation` is going to make sense now. It is generating values in states where the state will lead to the win. So if it leads you to a win quickly, it get's a higher result. Lets look at the last generation, where the loop converged.
<img src="./images/value policy map - iteration 32.png"/>
In this image, you can see that the entire matrix is filled with values that correspond to the state (in respect to its win value). **What is good to see, is that we 0 values on the places of a hole. We want this, because it has 0 value to be in that state, because you then lose the game.** <br>

I think that we have talked enough now. Lets analyze! what do we think is the best route for our guy to take? The route with the biggest values! Let's draw that onto the screen.
<img src="./images/value policy map - expected path.png"/>
If we take the highest value in the value table, then this is the path we want to take to the finish. But does our agent actually do this? You can see that in the video below.

<video width="100%" height="100%" controls>
  <source src="frozen lake result.mp4" type="video/mp4">
    Your browser does not support the video tag.
</video>

The agent is doing exactly what we thought it would. Nice to see that it does work. I think this is one of the easiest solutions of solving this problem in reinforcement learning.

## Further suggestions
This solution is not really smart, so it only uses values in the current state (without the possibility to adapt on random changes in the state) We can further investigate what solutions are smarter, and then see which one is the best solution to fit this problem (without over-engineering). But I will leave that to the next person creating an example.
