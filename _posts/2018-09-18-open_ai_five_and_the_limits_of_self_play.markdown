---
layout: post
title:  "OpenAI Five and the limits of self-play"
date:   2018-09-18 -0500
---
A few weeks ago, OpenAI attempted a new major milestone in AI development, a (nearly) full game of Dota2 against some of the best human players. 
Although the OpenAI Five was defeated by both of its professional opponents, the level of play was high and at times the match looked fairly even. 
This is amazing as the full game of Dota2 is very complex. 
Even more incredibly, the agent was trained using a relatively simple and very general reinforcement learning algorithm, [PPO](https://blog.openai.com/openai-baselines-ppo/). 

While the network structure has many bells and whistles to incorporate the complexities of the game, the algorithm itself is general enough to be applied to [robotics](https://blog.openai.com/learning-dexterity/), [image recognition](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zoph_Learning_Transferable_Architectures_CVPR_2018_paper.pdf), and many more tasks.
Congratulations to OpenAI and a huge win for RL!

{% include caption_img.html url="/assets/img/2018-09-18-open_ai_five_and_the_limits_of_self_play/network_diagram_08_06_2018.png" width="1000" description="Credit: <a href='https://blog.openai.com/openai-five-benchmark-results/'>OpenAI</a>"%}

The algorithm was used in conjunction to what is now a pretty popular trick of self-play. 
In environments that are competitive and easy to simulate, self-play refers to the agent learning purely by playing against itself. 
This way, simulations can be run very fast on thousands of CPUs/GPUs and years of experience collected every hour (256 GPUs and 128,000 CPUs for OpenAI Five). 
Self play was also used in the training of [AlphaGo Zero](https://deepmind.com/blog/alphago-zero-learning-scratch/).

Here, I have jotted down a few quick thoughts on why this extremely useful simulation trick of self-play puts a limitation to the performance of the agent. 
I will argue that while some of the deficiencies of the bots may be able to be quickly “trained away”, there are fundamental weaknesses in self play that will limit performance even after a lot of training. 
OpenAI Five may eventually beat the best human players, but as we make environments more complex, these issues will become more and more apparent.

My comments are only about the learning strategy on a very high level and not game commentary as I am not a Dota player myself. 
For in-depth commentary, I found [this](https://twitter.com/Smerity/status/1032810003466350592) to be great. 
Also, there is a [ton](https://blog.openai.com/openai-five/) [of](https://blog.openai.com/openai-five-benchmark-results/) [material](https://blog.openai.com/the-international-2018-results/) on this topic by OpenAI themselves and you can watch the games yourself [here](https://www.twitch.tv/videos/300508024?t=07h55m16s) and [here](https://www.twitch.tv/videos/300907006?t=07h44m03s). 

### Playing against oneself.
OpenAI five is really good at a single playing style. 
That is why it is often so hard to beat the [first time](https://blog.openai.com/dota-2/).
But in competitive games, once you figure out the opponent’s strategy, you can make it less effective.
Human players can do this by observing bot behavior during the matches and planning counters.

OpenAI Five, on the other hand, simply rolls out its learnt policy during human matches.
The policy takes into account the parts of the opponent's state that are visible to it and hence can react to developments in the game, but there is no attempt to update model weights based on human play.
So it is unable to react to human opponent's meta-strategy.

In gaming, the meta-strategy (or just meta) is how the game or specific heroes are generally played at a higher level than reactions to in-game developments. 
OpenAI's original [1v1 bot](https://blog.openai.com/dota-2/) was initially more powerful than a pro-gamer, but human players quickly identified its playing style and developed many counter metas within the [same event](https://www.reddit.com/r/DotA2/comments/6t8qvs/openai_bots_were_defeated_atleast_50_times/).

Self-play could overcome this if, during training time, the agent was pushed to encounter all different metas and hence forced to develop a single optimal policy which counters them all.

### Agent randomization
A standard way to do this in self-play is to sometimes play against an earlier version of oneself. 
The idea is that this will provide enough variation in opponents to avoid overfitting to itself.
OpenAI Five plays 80% of its games against itself and 20% against a former version of itself.  

This works well for stabilizing training, but is not the complete solution. 
The only opponent the agent has seen at par or better than itself is its own (future) policy.
Previous versions of the agent will be weaker than the current version.
Additionally, it is unlikely to see all good metas that exist in Dota2 as it is so complex and gradient descent training progresses along a single path which depends on the random seed of the network and the environment.
Moreover, metas change over time as players discover better ways to play the game and the game itself is updated (unlike Go which has remained unchanged for centuries).

In short, self play does not provide the agent enough variance in advanced metas to counter all strategies human players can form against it in a sufficiently complex environment.

This shows up in the value estimates of the trained agent.
OpenAI Five’s value estimates were remarkably good for the benchmark team.

> After the game 1 draft, OpenAI Five predicted a 95% win probability, even though the matchup seemed about even to the human observers. It won the first game in 21 minutes and 37 seconds. After the game 2 draft, OpenAI Five predicted a 76.2% win probability, and won the second in 24 minutes and 53 seconds. - [OpenAI](https://blog.openai.com/openai-five-benchmark-results/)

This is perhaps because the benchmark team played at a level below OpenAI Five and with a meta it has seen.

But for the pro team, the initial estimates of winning were optimistic despite the end result, perhaps because they employed a play style the agent never encountered during training and hence has an inaccurate estimate.
> ... [OpenAI Five] maintaining a good chance of winning for the first 20-35 minutes of both games. - [OpenAI](https://blog.openai.com/the-international-2018-results/)

<br>
<div style="border: 2px solid #CFB53B; padding: 10px;">
<span style="font-weight:bold">Side note or "There is no I in Fve":</span> 
OpenAI Five uses a bunch of <a href="https://gist.github.com/dfarhi/66ec9d760ae0c49a5c492c9fae93984a">reward shaping</a>, something that can be easily misused as warned by <a href="https://blog.openai.com/faulty-reward-functions/">OpenAI themselves</a>. 
One of the shaped rewards is called Team Spirit. 
It is a parameter that incentivizes agents to maximize overall team reward rather than just personal reward as the training progresses. 
One of the human strategies that led to OpenAI Five's downfall against pro team <a href="https://liquipedia.net/dota2/PaiN_Gaming">paiN Gaming</a> was to supercharge a single player, in this case hfn, who then was able to carry the team in later stages.
It is possible that OpenAI Five has never encountered this strategy in its training as Team Spirit encourages each individual hero to be less selfish about its own gains and OpenAI Five has never learned from an opponent without Team Spirit.
</div>
<br>

OpenAI themselves have employed a partial solution to this kind of overfitting in an earlier project by training ensembles of agents in parallel and playing against all of them.
Read more about it [here](https://blog.openai.com/competitive-self-play/#overfitting).
It is not mentioned whether this was done for OpenAI Five as well.

### Domain Randomization

But OpenAI did a bit more than play against earlier versions of itself. They employed a trick called domain randomization that has also been successful in other applications ([1](https://blog.openai.com/learning-dexterity/) [2](https://arxiv.org/pdf/1703.06907.pdf)).  

{% include caption_img.html url="/assets/img/2018-09-18-open_ai_five_and_the_limits_of_self_play/domain_randomization.png" width="600" description="Domain Randomization involves randomly varying aspects of the training environment, such as colors or physics, to make the network more robust. Credit: <a href='https://arxiv.org/pdf/1703.06907.pdf'>Tobin et al.</a>"%}

Examples of randomizations used by OpenAI Five are increasing/decreasing a hero's speed or starting health, assigning lane's randomly by providing shaping rewards etc.
These randomizations make the training even more robust by presenting an unseen play style, forcing the agent to explore more of its state space.
But this is not the same as a directed meta that is perhaps made to counteract its own.
Human play can exhibit a mode that is very far from the uniform sampling that domain randomization provides. 

### Solution
A solution could be a fast moving model of the meta play that updates according to opponent strategy. 
Data from professional human matches can be used to learn this fast meta layer and allow the agent to predict and quickly adapt to the style of play being used by humans. 
This could also be used to construct a domain randomization model that goes beyond just perturbing the physics or graphics and randomizes between entirely different human developed metas. 

Pure self-play as applied to OpenAI Five is blind to the problem of having to learn high level strategies from just a few samples, such as a single game of Dota2.
OpenAI Five plays centuries of games against itself every day, so a single game against humans will hardly make a difference to network parameters.
But humans are really good at this, which is why they are able to counter OpenAI Five after observing its playing style.
It is an essentially skill for AI that wants to compete against, or hopefully, work with us.
Self-play combined with such a strategy could be very powerful in learning competitive games.

<!--### There’s no I in Fve.-->

<!--A quick note about reward shaping in OpenAI Five. -->
<!--A puritan’s reward funciton for Dota2 would be +1 for winning the game and 0 everywhere else (or -1 for losing).-->
<!--But would be extremely hard to learn with as the reward would be so infrequently administered in the beginning. -->
<!--So a complex [reward structure](https://gist.github.com/dfarhi/66ec9d760ae0c49a5c492c9fae93984a) is employed to assist training OpenAI Five. -->
<!--This injects human bias into the game, such as laning early on in the game, cooperative play, etc.-->
<!--OpenAI themselves have warned about the perils of over-engineering shaping rewards [here](https://blog.openai.com/faulty-reward-functions/).-->

<!--One of the reward shaping strategies is called Team Spirit. -->
<!--It is a parameter that incentivizes agents to maximize overall team reward rather than just personal reward as the training progresses. -->
<!--This is partly to avoid having a communication strategy, as all heroes are controlled by different networks (with similar architectures but I am not sure if same weights).-->

<!--[>This is in lieu of a communication strategy as all 5 heroes are controlled by different networks but with similar architectures (I am not sure if weights are shared across heroes). <]-->

<!--This seems like a great idea at first but it precludes the agent from adopting a strategy similar to what the profession esports team, paiN gaming, did in their game against OpenAI Five.-->
<!--They supercharged one player, hfn, who then was able to carry the team in later stages contributing to the win. -->
<!--Since OpenAI Five is somewhat forced to maximize overall team reward in later stages of training, this strategy will be of extremely low value to it. -->
<!--Humans, on the other hand, are individual agents cooperating through communication, which leaves room for more individualistic play.-->
<!--This would not be a big hinderence to developing such a meta if the reward function was a binary win/loss.-->

<!--*Special thanks to [Ashley Edwards](https://www.cc.gatech.edu/~aedwards/) for her valuable inputs*-->
<!--<span style="font-weight:bold;"></span> -->
