1. Visualise the data

Plotted the (x, y) points to understand their general shape and range.

This helps in determining if the given model formula fits well.

2. t value assignments

Because there isn't a "t" column in this dataset, I assumed that the 1500 data points are uniformly sampled between t = 6 and 60.

3. Defining the model

Following are direct Python implementations of the given equations for the computation of predicted x and y values for any given (θ, M, X).

4. Choosing the loss metric

I used L1 loss (sum of absolute errors) instead of L2 because:

It is less sensitive to outliers.

It corresponds to the assessment method of the assignment.

5. Optimization strategy

The objective is to find the best parameters:

Grid Search – tested θ, M, X over coarse intervals within their allowed ranges.

Random Search – randomly sampled values near the best grid result.

Nelder–Mead Optimization: used scipy.optimize.minimize() for final fine-tuning.

6. Validation

After optimization, I plotted both:

the real vs predicted curve, and

the per-point L1 error over t. This helped confirm that the fitted parameters indeed matched the data.

https://www.desmos.com/calculator/qfrv75cfgc
