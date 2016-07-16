---
layout: post
title: Cartpole&#58; For Newcomers to RL
---

So here is my foray into RL research.  I was reading the 'Requests For Research' page on OpenAI and decided that it 
would be fun to try and work on some of the intro/warmup problems to see if I can do them.  Therefore I am starting on 
the [Cartpole: For Newcomers to RL](https://openai.com/requests-for-research/#cartpole) section.  Agreeing with their 
description of why the Cartpole environment is ideal for noobs here are a few reasons I chose to work on it:

1. The state space is low-dimensional, 4 dimensions in fact and that defines a Markov state
2. Simple action space - there are only two possible actions to choose from
3. Fast to run.  The simple linear agent I'll be building for this along with the simplicity of the environment means 
   I can actually run experiments in a reasonable amount of time on my CPU since I have no GPU available to me at present

To start off, as recommended by OpenAI, I used one of the simplest agents possible.  A linear agent.  The agent had four tunable parameters, otherwise known as weights ($$w$$) and these were combined with the state space $$s$$ at each timestep to produce a resulting action $$a$$.  First, the weights were multiplied by the state.  Remember, there were four weights and four state variables so the weights $$w = [w1, w2, w3, w4]$$ were multiplied by the state $$s = [s1, s2, 3, s4]$$ resulting in $$s*w = [s1w1, s2w2, s3w3, s4w4]$$.  Then, the elements of the vector were summed.  If the sum is less than zero, the agent  chooses action 0 (move the cart left) while if the sum is more than or equal to zero, the agent chooses action 1 (move the cart right). 

Sticking to OpenAI's laid out practice problems, I attempted to implement one of the simplest possible solutions: generating a larger number of completely random guesses at the 4 parameters.  This actually makes sense for a model with so few parameters.  We can guess four numbers right?  How many possible options could there be?  Well - infinite of course, we're talking about a continuous variable space [^1].  A more interesting question might be 'What percentage of all possible weights are could be considered a solution?'  To answer this we of course first need to define what we consider a solution.  I’ve chosen to use a slightly modified version of the OpenAI definition for the cartpole-v0 problem.  That being that the system is considered solved if the agent can balance the pole for at least 195 time steps on average over 20 episodes, where each episode ends at 200 steps or whenever the pole falls over.

Running some simple tests it became evident that a surprisingly large number of randomly selected parameters actually worked!  Around 17% stabilised the pole, holding it upright for more than 200 time steps.  This encouraged me to look a little deeper.  Instead of just picking random values I decided to perform a grid-search. I tested with a search space from -5.0 to +10.0 with 20 different values for each parameter.  This gave a 4 dimensional result with $$20^4$$ (160 000) entries.  I also upped the stopping limit from 200 steps per episode to 500 steps per episode.  The results were quite interesting.  A significant 17.77% of answers ‘solved’ the problem, attaining greater than or equal to 195.0 average reward over 20 episodes.  Interestingly, only 7.6% achieved an average reward of more than 400.  Also, 3.9% of all results achieved $$\geq$$ 499 reward.  The graph below shows the probability that an agent achieved less than the specified level of reward.  Ie according to the graph, the probability of an agent achieving an average reward per episode of 200 or less is around 82%.  Similarly, this means that the probability of achieving more than 200 reward per episode on average is around 18%, as mentioned previously.

![Cartpole Gridsearch Performance]({{ site.baseurl }}/images/blogs/cartpole_newcommers/Cartpole_random_performance_-5_to_10_cumulative.png)

### Hill climbing
After playing around with the random and grid-search solutions I decided it was time to move on to OpenAI’s next challenge in the series.  This was to implement a simple hill climbing algorithm for the cartpole problem.  The algorithm works as follows.

1. Start with a random guess at the parameters & evaluate performance.
2. Add a small amount of random noise to the parameters & evaluate again.
3. If the model gets better (average reward improves), then use the new parameters for the model, if it performs worse, throw the new parameters away.
4. If we achieve an average reward of at least 195.0 per episode then we’ve found a solution, otherwise go back to step 2.

There were three meta-parameters to this search and they were chosen as follows

* The initial starting position for each hill climb was chosen randomly, the parameters were again chosen from a uniform random distribution from -5 to 10.
* The maximum random step size (the small amount of noise added to the parameters) was set to ±1.0.  So the parameters couldn’t change by more than 1.0 in any direction in any single step.
* The maximum number of steps before giving up was set to 10000.  So if convergence was not achieved within 10000 steps we stop and count the run as a failure.

The search was performed 5421 times, each time starting from a different random position.  The average number of steps to convergence using this method was 16.2 though that’s not a representative metric.  In reality it did better than this.  A number of results never converged while others that took a very long time to.  This makes the averages steps to convergence skewed towards being worse than it practically would be.  I’ll explain this in more detail below but for now I’m culling the results to runs that converged in less than 150 time steps.  This reduces the total number of runs to 5404 and the average steps to convergence is now 7.7. 

That’s pretty good, but comically, it’s not actually as good as our random search.  On average the random method took $$\frac{1}{P(reward\hspace{1mm}\geq\hspace{1mm}195)} \approx \frac{1}{0.1777} \approx 5.63$$ steps while the hill climbing algorithm took an average of 7.7 steps.  We’ll ignore this for now and presume that the hill climbing algorithm would perform better than random search in a more complicated system.
Below is a histogram of the number of iterations the hill climbing algorithm took to converge.

![Hillclimb Aggregated Results]({{ site.baseurl }}/images/blogs/cartpole_newcommers/hillclimb_results_aggregated_iterations_all.png)

It’s of interest that on average, it took this algorithm 3.3 improvements to find a solution[^2].  From the histogram above it’s clear that a lot of this is due to the initial conditions often being quite favorable (perhaps already at a solution on the first step!).  A histogram of the number of improvements before convergence is shown below.

![Hillclimb Improvements Before Convergance]({{ site.baseurl }}/images/blogs/cartpole_newcommers/hillclimb_results_aggregated_iterations_improve_steps.png)

Now onto what I thought was the most interesting finding from the hill climbing results.  Remember those outliers that we removed?  Well now we’ll take a closer look at them.  In particular we’ll look at the three examples that didn’t converge in the 10000 step limit.  The question I’m interested in is “Why didn’t these examples converge?”.

Below is an image of one of the runs that did not converge (left), shown next to a run which did converge (right).

:-------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------:
![]({{ site.baseurl }}/images/blogs/cartpole_newcommers/testcart_hillclimb_results_4879.png)|![]({{ site.baseurl }}/images/blogs/cartpole_newcommers/testcart_hillclimb_results_4881.png)

It’s still not particularly clear why the left example didn’t converge.  Let’s take a look at the actual weights for the non-converging case.

```
      W0          W1          W2          W3       Reward
 0.71656432  0.61301802  0.94791112  0.12780639 10.00000000
 0.92818581  1.16844408  1.84387415 -0.70063127 10.26315789
 1.33728953  1.65332034  2.30747526 -0.92082660 10.31578947
 1.29997897  1.58504743  2.58488983 -0.75689915 10.36842105
 2.19948196  2.46740410  2.60304920 -1.40077723 10.47368421
 1.92880390  2.42471902  2.81164267 -0.49253086 10.52631579
 2.03341527  2.96348826  3.71440314 -1.28057189 10.57894737
 2.12329789  3.75464610  3.84856902 -1.00896565 10.63157895
 1.54490720  3.97368560  4.07093350 -0.02141334 10.68421053
```

Here the first 4 columns are the weights W0 to W3 and the last column is the average reward.  It can be seen that parameters W0 and W3 are moving fairly randomly while parameters W1 and W2 are both consistently increasing over time.  The result isn’t improving much, and the position is really just jiggling around on part of a five-dimensional manifold that seems essentially flat.  The reason we’re getting this behaviour is possibly due to the very low gradients in this region of the manifold.  It actually gets even worse in our case.  The manifold at this point is very noisy because we’re only averaging over 20 episodes.  If we averaged over a thousand or a million then the manifold would be much more smooth (though it would take a lot longer to run the simulation).  

Let’s pretend for now that the manifold in this region is completely flat (ie the gradient = 0, which it’s actually not).  The result from an evaluation is the average over 20 episodes and since the environment is stochastic, our example has some variance from run to run.  Now, even when we pick new parameters (ie move to a different position on the manifold), because our manifold is ‘flat’, we sample the result from the same distribution.  Since we’re taking a maximum over time, the longer we wait, the less likely we are to find a better solution.  This can be verified by looking at the time between updates in this flat region.  The first update occurred at step 2 giving 2 steps for an update, then the next was 4 steps later, then 11, then 5, 20, 80, 456 and 7040.  Because of this behaviour, the hill climbing algorithm got stuck out on a flat region of the manifold by initially moving in the ‘wrong’ direction due to random chance and the stochastic nature of the process.  Let’s look at some pictures to make this clearer.

#### W0W1
![Suck Hillclimb W0W1]({{ site.baseurl }}/images/blogs/cartpole_newcommers/stuck1W0W1.gif)

#### W0W2
![Suck Hillclimb W0W2]({{ site.baseurl }}/images/blogs/cartpole_newcommers/stuck1W0W2.gif)

#### W0W3
![Suck Hillclimb W0W3]({{ site.baseurl }}/images/blogs/cartpole_newcommers/stuck1W0W3.gif)

These animations show the run and how it progressed.  The line represents the path the hill search algorithm took and the dot is the position at some point in time.  The line shows only the improvements made by the hill climbing algorithm, not every random step.  When the run started, the position was quite close to a solution however due to random chance, the algorithm found an improvement on its position which was further out on the plateau.  Taking a look at the first image it’s obvious that no amount of moving along W0 or W2 was going to result in a solution.  The second image shows that for the entirety of the search, we weren’t actually that far away from a solution.  If we’d moved in the direction of positive W3 we’d have found one.  Unfortunately, the small gradient in W1 causes the search to move slightly in the positive W1 direction.  This consequently pulls the ‘cliff’ in the W0, W3 graph further away from the search location.  Ultimately, the search ends up in a position that is too far from any significant gradient to reach in a single step.  Therefore it will take a very very long time to reach a solution, if that’s even possible.

This has already been a fairly long post though I'd like to explore the use of gradient based methods such as SGD or policy-based learning methods. Perhaps that's for the next post.  
This is my first blog post on the subject so if you've got any feedback on the math, the science or just my writing style I'd love to hear it.  Please let me know in the comments!


### Footnotes:
[^1]: Ignoring that we're actually capped by the float precision we use.
[^2]: The algorithm usually performs many steps before finding an improvement but here we refere to just the number of improvements taken to find a solution.
