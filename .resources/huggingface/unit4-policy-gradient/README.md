# value-based methods, where we estimate a value function as an intermediate step towards finding an optimal policy.
# In value-based methods, the policy (π) only exists because of the action value estimates
#   since the policy is just a function (for instance, greedy-policy) that will select the action with the highest value given a state.
# With policy-based methods, we want to optimize the policy directly without having an intermediate step of learning a value function.

# The main goal of Reinforcement learning is to find the optimal policy π∗
#   that will maximize the expected cumulative reward. Because Reinforcement Learning is based on the reward hypothesis: all goals can be described as the maximization of the expected cumulative reward.

# Consequently, thanks to policy-based methods, we can directly optimize our policy πθ
# to output a probability distribution over actions
# π(a∣s) that leads to the best cumulative return. To do that, we define an objective function J(θ),
# that is, the expected cumulative reward, and we want to find the value
# θ that maximizes this objective function.

# The difference between policy-based and policy-gradient methods
# Policy-gradient methods, what we’re going to study in this unit, is a subclass of policy-based methods.
# In policy-based methods, the optimization is most of the time on-policy since for each update,
# we only use data (trajectories) collected by our most recent version of πθ
# The difference between these two methods lies on how we optimize the parameter θ:
# - In policy-based methods, we search directly for the optimal policy. We can optimize the parameter θ indirectly
#   by maximizing the local approximation of the objective function with techniques like hill climbing,
#   simulated annealing, or evolution strategies.
# - In policy-gradient methods, because it is a subclass of the policy-based methods, we search directly
#   for the optimal policy. But we optimize the parameter θ directly by performing the gradient ascent
#   on the performance of the objective function J(θ).

# Advantages over value-based methods
# 1. The simplicity of integration: estimate the policy directly without storing additional data (action values).
# 2. Policy-gradient methods can learn a stochastic policy: while value functions can’t.
# - We don’t need to implement an exploration/exploitation trade-off by hand. Since we output a probability distribution over actions,
#   the agent explores the state space without always taking the same trajectory.
# - We also get rid of the problem of perceptual aliasing. Perceptual aliasing is when two states seem (or are) the same but need different actions.
# 3. Policy-gradient methods are more effective in high-dimensional action spaces and continuous actions spaces
# - The problem with Deep Q-learning is that their predictions assign a score (maximum expected future reward) for each possible action
# - But what if we have an infinite possibility of actions? We’ll need to output a Q-value for each possible action!
# - And taking the max action of a continuous output is an optimization problem itself!
# - Instead, with policy-gradient methods, we output a probability distribution over actions
# 4. Policy-gradient methods have better convergence properties
# - In value-based methods, we use an aggressive operator to change the value function: we take the maximum over Q-estimates
# - Consequently, the action probabilities may change dramatically for an arbitrarily small change in the estimated action values if that change results in a different action having the maximal value.
# - On the other hand, in policy-gradient methods, stochastic policy action preferences (probability of taking action) change smoothly over time.
#
# Disadvantages
# - Frequently, policy-gradient methods converges to a local maximum instead of a global optimum.
# - Policy-gradient goes slower, step by step: it can take longer to train (inefficient).
# - Policy-gradient can have high variance. We’ll see in the actor-critic unit why, and how we can solve this problem.


# But how are we going to optimize the weights using the expected return?
# The idea is that we’re going to let the agent interact during an episode. And if we win the episode, we consider that each action taken was good and must be more sampled in the future since they lead to win.
# So for each state-action pair, we want to increase the
# P(a∣s): the probability of taking that action at that state. Or decrease if we lost.



