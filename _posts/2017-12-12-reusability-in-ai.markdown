---
layout: post
title:  "Learning to Compose Skills"
arxiv: "https://arxiv.org/abs/1711.11289"
github: "https://github.com/himanshusahni/ComposeNet"
date:   2017-12-14 -0500
---

One of the weaknesses of vanilla deep reinforcement learning is that policies and values learned are typically limited to a single environment, the one the agent was trained on. 
In other words, it is hard to transfer policies from one setting to another. 
This is in sharp contrast to how humans learn to do stuff. 
We draw heavily on past experiences and quickly learn what combination of skills we already have that works well in a new environment. 

A canonical environment in RL is the taxi cab domain. 
Let's say I train an agent to pick up and drop off a single passenger. 
Now I add another passenger to the mix. 
Instead of learning that it simply needs to execute the two skills, pick up passenger and drop off passenger, twice, the agent will move towards a brand new local minima for the two passenger problem. 
Once it has converged to that, it will forget how to do the single passenger case optimally. 
So, for every new problem, we end up initializing a new agent with random weights and train it to solve that problem only.  

{% include caption_img.html url="/assets/img/2017-12-12-reusability-in-ai/taxi-cab.png" width="400" description="Wut?" %}

Retraining from scratch for every new task is ridiculously inefficient. 
It is like building a very expensive rocket booster and only using it for a single flight. Who does that? 

{% include caption_img.html url="/assets/img/2017-12-12-reusability-in-ai/spacex-landing.jpg" width="600" description="Not these guys" %}

We need agents that learn to build on top of existing skills quickly in new, unseen environments. 
These agents should have very modular architectures, so one can plug in different skills and create complex hierarchies.

<br>
<div style="border: 2px solid #CFB53B; padding: 10px;">
<span style="font-weight:bold">Quick note:</span> I am certainly not the first one to think of this idea. 
There has been lots of work around this, notably in hierarchical and multi-task RL [1,2,3,4,5,6].
For a detailed description of related work and how this one fits in, please take a look at our <a href="https://arxiv.org/abs/1711.11289">Arxiv paper</a>.
Here, I will say that our focus is on <span style="font-style:italic">composing</span> already learned skills in a <span style="font-style:italic">variety</span> of interesting ways, instead of <span style="font-style:italic">decomposing</span> tasks into a <span style="font-style:italic">sequence</span> of sub-policies, which is found more commonly in literature.
</div>
<br>

Ok, so how do we go about obtaining such reusability of basic skills? 

Let's say we have pre-trained policies for skill 1, $$\pi_1$$, and skill 2, $$\pi_2$$, and we want a policy for a task that requires composition of both skills, $$\pi_c$$. 
Let's take an example of a gridworld with three objects, red, green and blue. 
$$\pi_1$$ can be "collect the red object" and $$\pi_2$$ can be "evade the green enemy" (an enemy object chases the agent at every step).
The composed policy, $$\pi_c$$ would make progress towards the red object *while* evading the green enemy. 
Let's also say that $$\pi_c$$ can be represented as a function of $$\pi_1$$ and $$\pi_2$$.

$$\pi_c : f(\pi_1, \pi_2)$$

In regular hierarchical RL, $$f$$ is a switch that picks one of the two skills depending on the state. 
It acts as a meta-controller. 
Run away if the enemy is too close, or run to the food if not.
One could also hand-design a function that blends the two policies together depending on the task. 
But that requires effort from an expert that understands each task and how policies work. 

Ok, so what if we learned the function $$f$$ instead?
How would we do that? 

If $$f$$ is differentiable, we could pass the policy gradient from the final policy through it and adjust its parameters. 
But what about $$\pi_1$$ and $$\pi_2$$? 
They are also typically represented as neural networks with their own set of parameters in deep RL. 
Those we don't want to modify, as we want to keep reusing them in the future for other composed tasks.
Let's write our equation down again, this time with the parameters.

$$\pi_c : f(\pi_{\theta_1}, \pi_{\theta_2} \lvert \theta_c)$$

$$\theta_1$$ and $$\theta_2$$ are parameters of the individual skill policies, and $$\theta_c$$ are addditional parameters that do the composition.

But, the final probability distribtion over actions, policy recommendations, from each skill may not contain enough information to blend them together. 
They are recommedations from each skill individually, and do not say anything about the current state of the combined task. 
We would instead like to embed information about the state and the skill policy into a single layer of a network, and provide the embedding to $$f$$ to *learn a composition* as the task requires. So, the
final form of our composition function is

$$\pi_c : f(e_{\theta_1}, e_{\theta_2} \lvert \theta_c)$$

where $$e_{\theta_1}$$ is a function that generates embeddings for skill 1 and similarly for $$e_{\theta_2}$$. 

Fantastic! 
Now we can learn the function $$f$$ for any kind of composition, such as collect red object *while* evading green enemy, or evade green enemy *and* blue enemy, etc. 

But wait, where do the embeddings come from?

### Phase 1
We break down the learning of our skill policies as follows,

$$\underbrace{\pi_1(a \lvert s)}_\text{policy for skill 1} = \underbrace{\pi(a \lvert e_1; \theta_p)}_\text{policy layer}  \underbrace{p(e_1 \lvert s; \theta_1)}_\text{skill-state embedding}$$

In other words, there are two networks, one that learns a state embedding for each skill, and the other outputs a policy given an embedding. 
The parameters for the latter, $$\theta_p$$, are shared across all skills.
In practice, this can be achieved using the following architecture.

{% include caption_img.html url="/assets/img/2017-12-12-reusability-in-ai/composenet-single.gif" width="800" description="Illustrative example of the forward and backward passes for a single skill" %}

This gif shows the training procedure for a single skill and the policy and skill modules. The black arrows represent the forward pass of generating an embedding and then the policy, and the red arrows denote the gradient.

{% include caption_img.html url="/assets/img/2017-12-12-reusability-in-ai/composenet-all.gif" width="800" description="Multiple skills are trained in parallel in separate environments."  %}

Each skill has its own network that generates embeddings given a state.
The policy layer takes any embedding and converts it to a distribution over actions which is executed in the environment. 
Each skills is running in its own separate environment, with its own reward function (+1 for successfully completing the skill, -1 for failing, with a small step cost).
Since the policy layer is shared, policy gradients from all skills are applied to it. 
But the embedding networks are trained using only gradients from the appropriate skill.

### Phase 2
This gives us a nice modular structure that allows us to do stuff like,

$$\pi_c(a \lvert s) = \underbrace{\pi(a \lvert e_c; \theta_p)}_\text{policy layer (fixed)        }   \underbrace{p(e_1 \lvert s; \theta_1)}_\text{skill 1 (fixed)} \underbrace{p(e_c \lvert e_1, e_2; \theta_c)}_\text{        compose embedding        }  \underbrace{p(e_2 \lvert s; \theta_2)}_\text{skill 2 (fixed)}$$

Or, the policy layer outputs a policy for the composed task given a composed embedding $$e_c$$, which is obtained by combining embeddings from skill 1 and skill 2. 
Note that the composition embedding parameters, $$\theta_c$$, is the only set of parameters that needs to be learned now, as the skill embedding and policy layer parameters have already been learned in phase 1 and are kept fixed.
Here is the above equation in practice.

{% include caption_img.html url="/assets/img/2017-12-12-reusability-in-ai/composenet-phase2.png" width="1000" description="Only the composition module needs to be trained for a new task"  %}

What's nice about this modular structure is we can construct arbitrary trees, or hierarchies, of skills and their compositions can be learned very quickly. 
For example, here is the composition for the task "collect the red object *while* evading the green enemy *and* the blue enemy", 

$$\pi_c(a \lvert s) = \underbrace{\pi(a \lvert e_c; \theta_p)}_\text{policy layer (fixed)        }   \underbrace{p(e_r \lvert s; \theta_r)}_\text{collect red skill (fixed)} \underbrace{p(e_c \lvert e_r, e_{and}; \theta_{while})}_\text{while composition}  \underbrace{p(e_{\neg g} \lvert s; \theta_{\neg g})}_\text{evade green skill (fixed)} \underbrace{p(e_{and} \lvert e_{\neg g}, e_{\neg b}; \theta_{and})}_\text{and composition} \underbrace{p(e_{\neg b} \lvert s; \theta_{\neg b})}_\text{evade blue skill (fixed)}$$


and the network,

{% include caption_img.html url="/assets/img/2017-12-12-reusability-in-ai/composenet-hierarchy.png" width="1000" description="Only the composition modules are trained"  %}

### Results

So how well does this architecture do? 
Here are graphs for some composed tasks. 
On the x-axis is always number of training steps. 
On the y-axis is either the average episodic reward or episode length over 50 evaluation runs.

<div class="container" style= "text-align: center; margin-bottom: 5pt;">
    <figure style= "display: inline-block;">
        <img src="/assets/img/2017-12-12-reusability-in-ai/Collect_2_Evade_1.png" width="350" />
        <figcaption style= "text-align: center;">Collect blue while evading green</figcaption>
    </figure>
    <!--<figure style= "display: inline-block;">-->
        <!--<img src="/assets/img/2017-12-12-reusability-in-ai/Evade_0_and_Evade_1.png" width="350" />-->
        <!--<figcaption style= "text-align: center;">Evade red and green</figcaption>-->
    <!--</figure>-->
    <figure style= "display: inline-block;">
        <img src="/assets/img/2017-12-12-reusability-in-ai/Collect_0_then_Collect_1.png" width="350px" />
        <figcaption style= "text-align: center;">Collect red then green</figcaption>
    </figure>
    <figure style= "display: inline-block;">
        <img src="/assets/img/2017-12-12-reusability-in-ai/Collect_0_Evade_1_Evade_2.png" width="350px" />
        <figcaption style= "text-align: center;">Collect red while evading green and blue</figcaption>
    </figure>
</div>

The orange line (ComposeNet) corresponds to learning compositions on top of skills. 
It always learns the task to near optimality. 
Hence, it is possible to learn to compose skills in this modular way and reuse them for many different tasks. 
It is also more efficient to do so than learn the task from scratch (solid blue line).
This is because ComposeNet only needs to train the composition layer to map two skills embeddings to an embedding for the composed task rather than learn everything from scratch.

Also shown is what happens if the pre-trained skills are provided as actions to the agent (metacontroller).
Initially we see a quick jump in average reward, but optimality is slow.
To understand this let's take the example of the "collect blue object while evading the green enemy" task.
Provided an option to get to the blue object, the agent quickly learns that some of the times it can get a large reward by making a beeline to it.
So it learns to spam a single action.
But it takes longer to get the more complicated control policy of alternating between running away from the enemy and moving towards the goal.
Moreover, this is not even the optimal policy.
The optimal behavior is blending the two skills together to make progress towards the goal *while* moving away from the enemy at the same time.

For more results, check out our [Arxiv paper](https://arxiv.org/abs/1711.11289)!

### Zero Shot Compositions

Can these functions learn specific compositions and apply them to unseen settings? 
For example, if I train a layer to do the *while* composition on all but one composed task, and test on the held-out task, will it generalize to it? 
The green line shows this composition "transfer" case.
It shows good zero-shot transfer (high rate of success with 0 training steps) and quick adaptation to the optimal policy for the new task.

### Are the skills really that important?

The advantage of this modular architecture we call ComposeNet is that for any new task, one only needs to train the composition layer to map embeddings from skills to the composed task.
But are the skills really that important?
What if we gave the network the wrong skills?
Will it still learn the overall task somehow by just storing the control policy in only the composition layer?
Or what if we simply re-initialize a new policy layer on top of one of the skills and retrain that to the composed task?

{% include caption_img.html url="/assets/img/2017-12-12-reusability-in-ai/Collect_0_Evade_1_wrongskills.png" width="1000" description="Collect red while evade green task with the incorrect skills."  %}

In the above graph, $$C()$$ denotes a composition function. 
The lines that just have the skills markings are for the case when a fresh policy layer is trained on top of that skill.
The task is to "collect the red object while evading green enemy".

This shows us a few things. 
Firstly, using any other skills than the correct ones does not perform as well. 

$$C(\text{red}, \neg \text{green})$$ is the case of using the skills "collect green object" and "evade red enemy", i.e. the opposite of what the task requires.
This performs nearly as well as the correct skills because you can invert the policy for collect to evade and almost invert the policy for evade to collect.
Given two completely irrelevant skills, "collect blue object" and "evade blue enemy", the task is not learned at all.
Retraining a policy layer on top of "collect red object" skill also gets close but not all the way to the best average reward.
Retraining a policy layer on top of "evade green enemy" skill does not solve the task because this skill says nothing about the red object which is necessary to get positive reward in this task.
This shows that the skill embeddings are encoding useful information about the objects they are concerned with and that composing the correct embeddings gives the best result.


### References
[1] T. G. Dietterich. Hierarchical reinforcement learning with the maxq value function decomposition. J. Artif. Intell. Res.(JAIR), 13(1):227–303, Nov. 2000.

[2] R. S. Sutton, D. Precup, and S. Singh. Between mdps and semi-mdps: A framework for temporal abstraction in reinforcement learning. Artificial Intelligence, 112(1):181 – 211, 1999.

[3] T. D. Kulkarni, K. Narasimhan, A. Saeedi, and J. Tenenbaum. Hierarchical deep reinforcement learning: Integrating temporal abstraction and intrinsic motivation. In Advances in Neural Information Processing Systems, pages 3675–3683, 2016.

[4] H. van Seijen, M. Fatemi, J. Romoff, R. Laroche, T. Barnes, and J. Tsang. Hybrid reward architecture for reinforcement learning. arXiv preprint arXiv:1706.04208, 2017.

[5] K. Frans, J. Ho, X. Chen, P. Abbeel, and J. Schulman. Meta learning shared hierarchies. arXiv preprint arXiv:1710.09767, 2017.

[6] A. S. Vezhnevets, S. Osindero, T. Schaul, N. Heess, M. Jaderberg, D. Silver, and K. Kavukcuoglu. Feudal networks for hierarchical reinforcement learning. arXiv preprint arXiv:1703.01161, 2017.

