https://huggingface.co/learn/deep-rl-course/

In the first unit, we saw two methods to find (or, most of the time, approximate) this optimal policy π∗.

In value-based methods, we learn a value function.
- The idea is that an optimal value function leads to an optimal policy π∗.
- Our objective is to minimize the loss between the predicted and target value to approximate the true action-value function.
- We have a policy, but it’s implicit since it is generated directly from the value function.
    For instance, in Q-Learning, we used an (epsilon-)greedy policy.

On the other hand, in policy-based methods, we directly learn to approximate π∗ without having to learn a value function.
- The idea is to parameterize the policy.
    For instance, using a neural network πθ, this policy will output a probability distribution over actions (stochastic policy).
- Our objective then is to maximize the performance of the parameterized policy using gradient ascent.
- To do that, we control the parameter θ that will affect the distribution of actions over a state.

actor-critic method, which is a combination of value-based and policy-based methods.
