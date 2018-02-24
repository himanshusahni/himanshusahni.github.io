---
layout: post
title:  "Reinforcement Learning never worked, and 'deep' only helped a bit."
date:   2018-02-23 -0500
---
**TL;DR:** RL has always been hard. Don't panic if the standard deep learning technique doesn't solve it.

[Alex Irpan's blog post](https://www.alexirpan.com/2018/02/14/rl-hard.html) does a great job laying out the many problems with current day deep reinforcement learning. 
But most of them aren't new --- they have always been there. 
In fact, they are the fundamental questions at the heart of reinforcement learning since its inception. 

With this post, I hope to show two things:
1. Most of the shortcomings described in Alex's post boil down to two core problems in RL, and
2. Neural networks only help us solve a small part of the problem, while creating some of their own.

<div style="border: 2px solid #0040ff;
       padding: 10px;">
<span style="font-weight:bold">Side note:</span> This post is in no way a rebuttal of Alex's claims.
On the contrary, I support most of his conclusions and believe that RL researchers need to be more transparent about the limitations of our approaches.
</div>

<!--### In the beginning was the MDP-->
<!--It is almost always formalized as an MDP, mostly because the math is just so convenient. -->
<!--An MDP consists of states, actions, transitions and rewards.-->
<!--The first three things tell the agent about the environment and its control over it.-->
<!--The last one tells it about its goal.-->

<!--{% include caption_img.html url="/assets/img/2018-02-17-reinforcement-learning-never-worked/agent-env.png" width="650" description="Perhaps the most popular image in RL [1]" %}-->

### The two main problems of RL

The highest level description of reinforcement learning is the maximization of some notion of long term return by acting in an environment. 
There are two fundamental difficulties one encounters while solving RL problems: the balance of exploration vs. exploitation and long term credit assignment.

As discussed in the first page of the first chapter of the reinforcement learning book by Sutton and Barto [[1]](http://incompleteideas.net/book/bookdraft2018jan1.pdf), these are unique to reinforcement learning.

There are closely related extensions to the basic RL problem which have their own scary monsters like partial observability, multi-agent environments, learning from and with humans, etc.
We're just gonna ignore all of that for now.

{% include caption_img.html url="/assets/img/2018-02-17-reinforcement-learning-never-worked/this_is_fine.gif" width="800" description="RL researchers all the time" %}

Supervised learning, on the other hand, deals with the problem of generalization.
Generalization is assigning labels to unseen data, given that you already have a bunch of seen data with their labels. 
Some parts of the fundamental problems in RL can be solved by good generalization.
If you could generalize well into unseen states, you wouldn't need to explore as much.
This is where deep learning usually fits in.

As we will see, reinforcement learning is a different and fundamentally harder problem than supervised learning. 
It is not so surprising if a wildly successful supervised learning technique, such as deep learning, does not fully solve all of the challenges in it.
In fact, deep learning, while improving generalization, brings with it its own demons.

What *is* surprising, is the surprise at RL's current limitations.
The inability of DQN to deal with long horizons, or the latest deep actor critic method taking millions of environment steps is not a new and mysterious addition circa deep reinforcement learning.
This happens because of the very nature of the problem and has always been present.

Let's go into detail on these two fundamental problems and it should be clear why it's not surprising that reinforcement learning doesn't 'work' yet.

### Exploration-vs-exploitation
##### Sample inefficiency, reproducibility, and escaping local optima

This is the question every agent must learn to answer from a very early age, do I keep following this policy that's giving me nice returns, or do I take some relatively sub-optimal actions now in case there's a possibly bigger payoff later?
This problem is so hard because there can be no right answer in general - there is always a trade-off. 

#### Off to a good start.

The famous Bellman equations only guarantee convergence to the optimal value function if every state is visited an infinite number of times and every action is tried an infinite number of times in it.
So right off the bat, we need an infinite samples to learn, and we need them everywhere!

You may say something like "Why obsess over optimality?"

Fair enough.
In most cases, a policy that achieves the goal, doesn't take too long, and doesn't mess up too many other things is fine.
Sometimes in practice, we are happy to find that a good policy can be learnt in a finite number of steps (20 million is much smaller than infinity).
But it is hard to define these subjective notions without attaching numbers to maximize/minimize something.
It is even harder to provide any sort of guarantees on it.
More on this later.

Ok so let's just say we are happy with an approximately optimal solution (whatever that means).
The number of samples needed to get the same approximation increases exponentially with the state *and* action space.

#### But wait, it gets worse.

Without any assumptions, there is no better way to explore than randomly.
You can add some heuristics, such as curiosity [[2]](http://people.idsia.ch/~juergen/interest.html), and they work well in some cases, but we do not have a complete solution yet.
After all, you have no reason to believe there's a bigger or smaller payoff behind any action in a particular state unless you try it.

What's more, model-free reinforcement learning algorithms typically try to solve the most general formulation of the problem.
As in, there are no assumptions about the form of the state distribution, the transition dynamics of the environment or of optimal policies (for eg. [[3]](https://arxiv.org/abs/1707.06347))

And this makes sense.
Just because you see a great reward once doesn't mean you will always get it every time you are in that state and take that action.
The only sensible thing to do then is to not trust any particular reward too much and only slowly make changes to your belief of how good an action is in a state.

Ok, so you are making small, conservative updates to functions that are trying to approximate expectations of arbitrarily complex probability distributions over an arbitrarily large number of states and actions.

#### But wait. It actually gets even worse.

Let's talk about continuous states and actions.

The world at our size seems to be mostly continuous.
But that's a problem for RL.
How are you supposed to visit an infinite number of states an infinite number of times and take an infinite number of actions an infinite number of times in them?
If only you could generalize some of the knowledge you have learned to unseen states and actions.
Supervised learning!!

Let me explain that a bit.

Generalization in RL is called *function approximation*. 
Function approximation captures the idea that the state and actions can be fed into a function that computes their value, instead of having to store the value of each state and action in a giant table.
Train the function using data and you're basically doing supervised learning.
Job done.

#### Not so fast.
Even this is not that straightforward in RL.

First, let's not forget that neural networks have their own exorbitant sample inefficiency due to the slow pace of gradient descent.

#### But wait, the situation is actually even worse.
<!--{% include caption_img.html url="/assets/img/2018-02-17-reinforcement-learning-never-worked/rl-phd.jpg" width="700" description="" %}-->

In RL, the training data for the network must be bootstrapped on-the-fly by interacting with the environment.
As you explore and collect more data, your estimates of the action-value, Q, keep changing.

Unlike supervised learning, the ground truth labels to your network are not fixed!
Imagine if at the beginning of Imagenet training you label an image as a cat but later kept changing your mind to dog, car, tractor, etc. 
The only way you can get closer and closer estimates of the true target function for your network is to keep exploring.

You are never actually given any samples from the true target function, which is the optimal value function or policy, even in your training set.
Yet, you are still able to learn!
*That* is the reason reinforcement learning is [popular](https://twitter.com/jacobandreas/status/924356906344267776).

So far, we have two very unstable things that must be slowly modified to prevent from totally collapsing.
Rapid exploration may bring about sudden changes in the target landscape, which the network is so painstakingly trying to fit to.
This double whammy of exploration and network training results in higher sample complexity than normal supervised learning problems.

Exploration in uncertain dynamics also explains why RL seems to be more sensitive to hyper-parameters and random seeds than supervised learning.
There are no fixed datasets your networks are training on.
The training data is directly dependent on the network output, whatever exploration mechanism you use, and environment randomness. 
Therefore, with the same algorithm on the same environment in different runs, you may see dramatically different training sets, leading to dramatically different performance (take a look at [[4]](https://arxiv.org/abs/1709.06560)).
Again, the core problem is that of controlling exploration to see similar distributions of states, something that the most general algorithms make no assumptions over.

<!--Let's not forget that the environment is never nice enough to provide you with independent and identically distributed samples.-->
<!--The very nature of exploration in sequential decision making problems is that the next sample you will see is the direct result of the current one.-->
<!--There are some tricks to get past this, and some other tricks to improve those tricks, but those are just to cover up for the fact that neural networks are just not capable of fitting correlated samples.-->

#### But wait! It gets even...

For continuous action spaces, the most popular methods are *on-policy*.
On-policy methods can only use samples consistent with the current policy that is being executed.
This also means that as soon as you update your current policy, everything you have experienced in the past becomes immediately unusable.
Most algorithms you hear of in the weird domain with orange humans and animals that look like a bunch of piping ([Mujoco](http://www.mujoco.org/)), are on-policy.

{% include two_img_caption.html url1="/assets/img/2018-02-17-reinforcement-learning-never-worked/cheetah.gif" width1="530" description1="A cheetah" url2="/assets/img/2018-02-17-reinforcement-learning-never-worked/pipage.gif" width2="300" description2="Piping"  %}

Off-policy methods on the other hand, can learn the optimal policy by observing any other policy being executed.
This is obviously much nicer but unfortunately we are just not that good at it yet.

#### But wait!

No, actually that's all I have for now.
It'll get worse in the next subsection though.

{% include caption_img.html url="/assets/img/2018-02-17-reinforcement-learning-never-worked/spacex-landing.jpg" width="600" description="This is gonna start looking easy"  %}

To summarize, these issues arise because of a core problem in reinforcement learning and more broadly in all of AI: exploration. 

RainbowDQN takes 83 hours to learn because it does not come preloaded with notions of a video game is, that enemies shoot bullets at you, that bullets are bad, that a bunch of pixels that seem to stay together is a bullet, that bullets exist in the world, that objects exist, that the world is organized into anything more than a maximum entropic distribution. 
All of these are priors that help us, humans, dramatically limit our exploration to a small set of high quality states.
DQN has to learn all of these by mostly random exploration.
That it learns to beat expert humans, and centuries of wisdom in the case of AlphaZero, is still very surprising.

<!--But it is also incentivized to achieve the optimal policy.-->
<!--Humans learn to play games in minutes but even after lifetimes (in fact millenia) of experience cannot beat a randomly exploring RL agent trained for a few days (AlphaZero).-->

<!--One of the points I strongly agree with from Alex's post is that of generalization beyond the current domain.-->
<!--I believe reusability of skills will unlock the next big gains in reinforcement learning.-->
<!--But just relying on the generalization power of neural networks is not going to be enough.-->
<!--For one it has been shown that beyond a certain amount of domains, the performance drops significantly as the network capacity saturates [?]-->

<!--Neither is meta-learning a solution because the whole point is to transfer from *unseen* settings.-->
<!--Building a prior by sampling from domains that are *very* similar to the ones you are going to be learning on destroys the whole purpose of using transfer in the first place.-->
<!--If you have already learned "Cheetah" at 0.5 speed, 2.5 speed, 0.9 speed, 2.2 speed, etc. then why are you learning it on 1.0 speed anyways?-->

<!--A much more promising approach is hierarchical reinforcement learning which I discuss below.-->

### Long term credit assignment
##### Reward functions, their design, and transfer

You know how some people will scratch their lottery ticket only with a lucky coin because one time they did and they won a lot of money?
RL agents are basically playing the lottery at every step and trying to figure out what they did to hit the jackpot.
They are maximizing a single number which is the result of actions over multiple time steps mixed in with a good amount of environment randomness.
Figuring out which series of actions are actually responsible for the high reward is the problem of credit assignment.

You want rewards to be easy to specify.
The promise of reinforcement learning is that you tell a robot when it has done something right and over time it learns how to do that thing reliably.
You don't have to actually know how to do the thing yourself and you don't have to provide supervision at every step. 

The problem actually occurs because the scale at which we can provide rewards for meaningful tasks is much larger than the scale current day algorithms can handle.
The robot is operating at a much faster time scale of what joint velocities it should set at every millisecond and the human is expecting to only reward it once it has made a good sandwich.
There are many decisions that happen in between and if the gap between the crucial choices and the reward is too big, any current day algorithm will just fail.

There are two solutions to this.
One is to reduce the scale at which rewards are provided, i.e. provide shaping rewards more frequently.
As usual, though, if you give an optimization algorithm a weakness, it will exploit it all the way to optimality.
If the reward is not well designed it can lead to [reward hacking](https://blog.openai.com/faulty-reward-functions/).

<!--Reward hacking means unexpected and undesired behavior which maximizes reward but does not reflect the true desires of the designer.-->
<!--Reward hacking has existed since time immemorial. -->
<!--Or at least way before OpenAI popularized it with the [boat racing video game example](https://blog.openai.com/faulty-reward-functions/).-->
<!--Although a video game is great for marketing the problem, it can be understood in a much simpler setting. (To be fair to Alex, he did mention many other examples as well).-->

<!--Suppose you want to train your agent to ride a bicycle [?].-->
<!--You reward it every time it makes forward progress without falling down.-->
<!--Suppose also that you decide to terminate the episode once it reaches the goal.-->
<!--Seems like a reasonable setup.-->
<!--But the agent will very quickly learn to just go in circles, approaching the goal then turn around at the last instance, then return to keep racking up the rewards.-->

{% include caption_img.html url="/assets/img/2018-02-17-reinforcement-learning-never-worked/rewards.jpg" width="600" description=""  %}

<!--Without a penalty for moving away from the goal, there is no incentive for the agent to actually end the episode by getting to the goal.-->
<!--These kinds of design errors are easy to make when the environment is more complicated and the interactions of different rewards is not easy to reason about.-->

Ultimately, we fall for this because we forget that the agent is optimizing in the value landscape, not for the immediate rewards.
So even if your immediate reward structure seems innocuous, the value landscape may be very non-intuitive and may have many of these exploits if one is not careful.

Which begs the question, why are rewards used in the first place?
Rewards are a way of specifying goals that will let us use the power of optimization to get a good policy.
Shaping rewards are a way to inject more domain specific knowledge on top.

Are there better ways of specifying goals?
In imitation learning, you can slyly sidestep the whole RL problem by asking for labels directly from the target distribution, i.e. the optimal policy.
There are other ways of learning without direct rewards [[5]](https://dl.acm.org/citation.cfm?id=1015430), or providing goals to agents as images [[6]](https://arxiv.org/abs/1608.03824).
(Stay tuned for an ICML workshop on Goal Specification in RL!)

Another promising way to deal with long horizons (hugely delayed rewards) is hierarchical reinforcement learning.
I was surprised when this did not make it into Alex's post because it is the most intuitively appealing solution to this problem (but I guess I'm [biased](http://himanshusahni.github.io/2017/12/26/reusability-in-ai.html)!)

Hierarchical RL attempts to decompose a long horizon problem into a series of goals and subgoals.
By decomposing the problem, we are effectively dilating the time scale at which decisions are being made.
The really exciting stuff is if the policies being learned for subgoals can also be applied to achieving other goals.

In general the hierarchy can go as deep as desired.
The canonical example is that of traveling to another city [[7]](http://www-anw.cs.umass.edu/~barto/courses/cs687/Sutton-Precup-Singh-AIJ99.pdf).
The first choice is to decide whether to go or not go.
Once that is determined, you have to decide how each leg of the journey will be completed.
Taking the train to the airport, flying, and then taking a taxi to the hotel seems reasonable. 
Taking the train then involves looking up schedules, purchasing tickets, etc.
Calling a taxi includes various motor actions to pick up a telephone and create vibrations in the vocal chords.

{% include caption_img.html url="/assets/img/2018-02-17-reinforcement-learning-never-worked/deeper.jpg" width="600" description="Legit RL research request"  %}

Although the example is a bit too simplistic, it does make the point in a quaint 1990s way.
The single scalar reward for reaching the desired city can be propagated all the way back through the Markov chain over these different levels of hierarchy.

The promise is great but we're not there yet on this either.
Most state of the art works only consider hierarchies a single level deep and the goal of transfer to different tasks is hard to achieve.

### Conclusion
My conclusion is mostly identical to Alex's.

I am very glad there is so much activity in the field and we are finally tackling the kinds of problems I've always wanted to.
Reinforcement learning has finally progressed beyond the gridworld!

{% include caption_img.html url="/assets/img/2018-02-17-reinforcement-learning-never-worked/tesla_roadster.jpg" width="600" description="Don't Panic!" %}

My only additional message is that do not despair if the standard deep learning techniques don't slay the monsters of reinforcement learning.
Reinforcement learning has two fundamental difficulties not present in supervised learning - exploration and long term credit assignment.
They have always been there and will take more than a really nice function approximator to solve.
We have to find much better ways to explore, use samples from past exploration, transfer across tasks, learn from other agents including humans, act in different timescales and address the awkward problems with scalar rewards. 

Despite the ridiculously challenging problems in RL, I still believe it is the best framework we have right now to work on general intelligence.
Otherwise, I wouldn't be in this field.
Watching DQN play atari from visual input, or AlphaGo defeating the world champion in Go, were truly moments in which we witnessed a small advance in general intelligence.

I'm excited by the future RL will bring for AI.


*Special thanks to [Ashley Edwards](https://www.cc.gatech.edu/~aedwards/) and Yannick Schroecker for the valuable inputs!*
<!--<span style="font-weight:bold;"></span> -->

### References
[1] Richard S. Sutton, and Andy G. Barto. Reinforcement learning: An introduction. Vol. 2. Cambridge: MIT press, 2017.

[2] Jurgen Schmidhuber. Curious model-building control systems. In Proc. International Joint Conference on Neural Networks, Singapore, volume 2, pages 1458-1463. IEEE, 1991

[3] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347 (2017).

[4] Peter Henderson, Riashat Islam, Philip Bachman, Joelle Pineau, Doina Precup, and David Meger. "Deep reinforcement learning that matters." arXiv preprint arXiv:1709.06560 (2017).

[5] Pieter Abbeel, and Andrew Y. Ng. "Apprenticeship learning via inverse reinforcement learning." In Proceedings of the twenty-first international conference on Machine learning, p. 1. ACM, 2004.

[6] Ashley Edwards, Charles Isbell, and Atsuo Takanishi. "Perceptual reward functions." arXiv preprint arXiv:1608.03824 (2016).

[7] Richard S. Sutton, Doina Precup, and Satinder Singh. Between mdps and semi-mdps: A framework for temporal abstraction in reinforcement learning. Artificial Intelligence, 112(1):181 – 211, 1999.
