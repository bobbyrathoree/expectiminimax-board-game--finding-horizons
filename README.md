# a2-2
## Part 1: IJK

## Non-deterministic IJK: 

In the Deterministic version of IJK, the new letter ‘a’ was added to the first empty tile which is found traversing from the top left corner of the board. Whereas in the non-deterministic version of IJK, the letter a is added randomly to any empty tile on the board. In this case, the letter could be placed at any empty tile of the current board with equal probability. To solve this problem we implemented expectiminimax algorithm with depth 5 (Max layer, expected layer, Min layer, expected layer, Max layer)  which takes the randomness into account.  

At each depth for Min and Max player, we perform Left, Right, Up, and Down moves and generate the successor boards by placing one letter A or a (according to the turn of players) on every empty tile of the current board configuration. For example, on the first board configuration we have ‘A’ placed on some tile. We make 4 copies of this configuration and perform Left, Right, Up, Down moves on the copies. To generate the successors, we create a list of free tiles and then create different configurations of boards with an a placed at the co-ordinates from the free tiles list. 

We created a heuristic function which used the property of gradience, number of free tiles, number of merges. During our discussions we concluded that we would want the all the higher valued tiles pushed to any one corner of the board. We also referred to this discussion here- https://stackoverflow.com/questions/22342854/what-is-the-optimal-algorithm-for-the-game-2048/22498940#22498940. The property of gradience ensures that the higher valued tiles are closer to each other and pushed in a corner and hence can be merged easily, and the lower valued tiles are away from the corner and the newly added tiles can be easily merged with the lower valued tiles. Free tiles count the number of empty spots on that successor board configuration. Initially when the game is just beginning, the number of free tiles have no weightage in the heuristic. Only when the board is more than half full that we would want to give weightage to number of free tiles on successor boards as we need free tiles for game to continue. We calculated the number of possible merges on the successor boards. High point are given for merges which result in higher letter values on board. We also tried giving penalties when a move resulted in the opponent having the highest valued tile on the board. 

The leaf nodes (successor boards at depth 0) are evaluated using the heuristic function. For the Min player, the expected value is calculated by taking the weighted average of the leaf nodes (as in this case each successor is equally probable). The Min player will select the minimum values from the expected nodes. Another expected layer on top of the Min layer takes the weighted average of the Minimum values selected by the Min player. The Max player selects the maximum value from the expected nodes. For this Max player, these expected nodes denote the L, R, U, D moves. 

Initially we when we started coding solution for this problem, we were passing ‘game’ to our function and performing L, R, U, D moves on it. Later, we realized what we were doing was modifying the same game object. And then we started using deepcopy of the game object instead. New lesson learned. Our implementation of expectiminimax takes slightly long time to find the best move for the player during the initial turns in the game when the board is relatively empty. 
## Part 2: Finding horizons

Here, we tackle one of the most classic problems in computer vision in which we need to identify where exactly a picture was taken on our planet. We focus on rather a subset of this problem here, with the assumption that if we're able to identify the horizon decently enough, we could use this as a fingerprint and match it with a digital elevation map to classify where a particular image was taken.

We're assuming here that the images we work/test on will have clear looking mountain ridges, with nothing blocking them, and that the sky is relatively clear. We need to "estimate" the row of the image corresponding to the ridge boundary and plot the estimated row to get our superimposed image.

### Bayes Net
We've already been give the code to calculate the edge strength map of a given image that simply measures the local gradient strength at each point. Using the naive bayes net algorithm is real simple: we just take the max of each column of each row at a particular instance using the argmax function provided by numpy. Some of the results are as follows:

![output 1](part2/bayes_net/mountain_output_simple.jpg)

![output_2](part2/bayes_net/mountain4_output_simple.jpg)

![output_3](part2/bayes_net/mountain7_output_simple.jpg)

Not bad, but... bad. What can we do here?

### Viterbi
We could try to find all the different scenarios of hidden states for the given sequence of pixels and then identify the most probable one. However, it will be an exponentially complex problem to solve. To get better results, we implement the Viterbi algorithm, that is a dynamic programming approach to solve the problem. We basically expand on the idea that at each time step we calculate, we only need to store the sequence path to the pixel that has the best probability going into each state. If our HMM has only 2 states for instance, we only need to store at most 2 paths, updated at every time step, because all that matters for the next time step is where we were at in the previous time step. That's basically it.
To implement that in code, it sure was tricky! We've implemented the viterbi algorithm in two rather similar ways, one that involves backtracking overtly, the other one not needing one due to memoization. One using log of the probabilities, while the other one uses the probabilities, like straight up. We kept both in the code and used one for the regular viterbi part and the other for the human feedback part that we'll come across in a second.

In the viterbi function that doesn't involve logs, we started out with normally distributed initial probabilities, followed by the emission probabilities that were the result of each pixel's value divided by the max of it's row. The transitional probabilities were the absolute value of each row normalized using it's respective max column values. The two numpy arrays first_product and second_product keep track of each 2 states along the way, after updating with the max of the previous column and the current column's transitional and emission probability. Then, we just backtrack using the two numpy arrays first_product and second_product to get out resultant_matrix, i.e. the y-coordinates to superimpose on top of the original image. Some sample outputs:

![output 1](part2/viterbi/mountain5_output_map.jpg)

![output_2](part2/viterbi/mountain4_output_map.jpg)

![output_3](part2/viterbi/mountain7_output_map.jpg)

Better. But can it be improved even further?

### Viterbi with human feedback

Now, in this case, everything else is same, except we're given a set of pixel coordinates that we can assume lie on the ridge of the mountain of a given picture. This should help reduce the state space for the algorithm to work on, since now we can assume that the ridge has to be near the given coordinates. We change the emission probabilities, as in change the column where the pixel lies to a value really close to zero and make that particular pixel equal to 1, since it is definitely part of the ridge. Before that, we normalize the emission probabilities using a similar technique that we used for the initial probabilities in the regular viterbi function. Sample outputs are as follows:

![output 1](part2/human_viterbi/mountain5_output_human.jpg)

![output_2](part2/human_viterbi/mountain8_output_human.jpg)

![output_3](part2/human_viterbi/mountain_output_human.jpg)
