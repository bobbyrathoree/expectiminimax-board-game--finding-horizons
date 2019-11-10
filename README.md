# a2-2

## Part 2: Finding horizons

Here, we tackle one of the most classic problems in computer vision in which we need to identify where exactly a picture was taken on our planet. We focus on rather a subset of this problem here, with the assumption that if we're able to identify the horizon decently enough, we could use this as a fingerprint and match it with a digital elevation map to classify where a particular image was taken.

We're assuming here that the images we work/test on will have clear looking mountain ridges, with nothing blocking them, and that the sky is relatively clear. We need to "estimate" the row of the image corresponding to the ridge boundary and plot the estimated row to get our superimposed image.

We've already been give the code to calculate the edge strength map of a given image that simply measures the local gradient strength at each point. 