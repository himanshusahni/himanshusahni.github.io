---
layout: post
title:  "Reusability in Artificial Intelligence"
date:   2017-11-12 -0500
tags: [Artificial Intelligence, Reinforcement Learning]
---
I believe that the reusability will be key to more general AI. Currently, if I train a Neural Network to identify dogs and cats, and I throw a panda into the mix, I will have to retrain the whole system from scratch with brand new weights. Or if in RL, I train an agent to pick up and drop of one passenger and I throw another passenger into the domain, instead of learning to simply sequence the two pickups/dropoffs in an optimal way, it learns a brand new local minima for the two passenger
case. It is ridiculously inefficient to retrain basic skills every time you want to add a new capability to an agent. It will be like building a very expensive rocket booster and only using it for a single flight. Who does that?

![No Waste](https://github.com/himanshusahni/himanshusahni.github.io/raw/master/assets/img/spacex-landing.jpg "Not these guys")
