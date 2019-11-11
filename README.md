# a2-2

## Part 2: Finding horizons

Here, we tackle one of the most classic problems in computer vision in which we need to identify where exactly a picture was taken on our planet. We focus on rather a subset of this problem here, with the assumption that if we're able to identify the horizon decently enough, we could use this as a fingerprint and match it with a digital elevation map to classify where a particular image was taken.

We're assuming here that the images we work/test on will have clear looking mountain ridges, with nothing blocking them, and that the sky is relatively clear. We need to "estimate" the row of the image corresponding to the ridge boundary and plot the estimated row to get our superimposed image.

We've already been give the code to calculate the edge strength map of a given image that simply measures the local gradient strength at each point. Using the naive bayes net algorithm is real simple: we just take the max of each column of each row at a particular instance using the argmax function provided by numpy. Some of the results are as follows:




We could try to find all the different scenarios of hidden states for the given sequence of pixels and then identify the most probable one. However, it will be an exponentially complex problem to solve. To get better results, we implement the Viterbi algorithm, that is a dynamic programming approach to solve the problem. We basically expand on the idea that at each time step we calculate, we only need to store the sequence path to the pixel that has the best probability going into each state. If our HMM has only 2 states for instance, we only need to store at most 2 paths, updated at every time step, because all that matters for the next time step is where we were at in the previous time step. That's basically it.
To implement that in code, it sure was tricky! We've implemented the viterbi algorithm in two rather similar ways, one that involves backtracking overtly, the other one not needing one due to memoization. One using log of the probabilities, while the other one uses the probabilities, like straight up. We kept both in the code and used one for the regular viterbi part and the other for the human feedback part that we'll come across in a second.