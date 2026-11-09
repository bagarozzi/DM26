# Business understanding
Important to put the man in the loop. Understand more data throught understading more the domain.

## Analysis techniques:
### Univariate and Bivariate
Single or double-variable analysis of their distribution and central tendency. In this process each variable is analysed on its own to understand key indicators and understand how the feature behave.

> This process is importnat for the next steps of *feature engineering* to understand the data's stability.

Statistics collected are: relative and absolute frequency, mean, mode, cardinality, missing values. Others are:
- Dispersion measures: variance, std. dev., range, median absolute deviation
- Shape of the distribution: skewness, kurtosis

**Skewness** is calculated as the average cube difference w.r.t. the mean, divided by the standard deviation cubed. Cubing preserves the sign so the skewness' sign tells us if there's a asymmetry in the distribution (w.r.t. the mean).
The presence of standard deviation enables comparing skewness of different distribution (how age dist. is skewed compared to income).

**Kurtosis** measures how the *tail* of a distribution is extended towards the outside. It's the same as skewness but to the 4th power, so there's no sign but large differences contribute heavily to the result. $K(x) = 3$ is the tail of the normal distribution.

Other information we can analyse is: **data granularity** (which is the level of detail of raw data), **data quality** (missing values, noise, duplicated).

**Bivariate analysis** studies the relationship between two variables (e.g. correlation).

**Pearson Correlation** is defined as the ratio of the covariance between two variables and the product of their deviation. This figure, between -1 and 1, measures the existence of a linear relationship. The presence of the standard deviation *normalises* the scale; at the same time, the contribution of outliers is large due to their effect on stddev.

>If two variables correlate, one of the two can be dropped out during feature selection/engineering because the information from one *can be derived from the other* it does not bring *any additional information*.

*Pearson's cannot detect other types of non-linear correlation.*

**Spearman correlation** is less sensible to outliers that Pearson's, it detects correlation between the **ranks** of two values measuring strenght and direction of a *monotonic correlation*, calculating the rank corrects the presence of outliers. So it's best used when correlation is not necessarily linear, stricly monotonic and in presence of outliers.

- ***Spurious correlation***: two attributes appear related but there's no direct link (usually there's a correlation with a third, transitive, common, link; bias, trend or aggregation).
In this case, fixing a value for the third variable (or choosing an interval/bucket) makes the correlation vanish; in time series is best to remove seasonality and trends to avoid this correlation.
- ***Mutual information***: measures how much one variable is predictable from the other. If $MI = 0$ then the variables are independent, otherwise there is some relationship (not necessarily correlation). 
It's calculated from joint probability and the product of the two probability -> *comparing the probability as if they were independent against the joint probability*; this is calculated for each $x,y$.

### Multivariate analysis
Analysis of variable in highly-dimensional spaces. PCA and t-SNE. When data becomes sparse it loses meaning, data points are all equally distant and much more data is required.

In **Principal Component Analysis** the solution is expressed in a new, **low-dimensional, space**. It assumes that the most information comes from **variables with the highest variance**. 
Mathematically, the co-variance matrix is computed and from it are selected $k$ eigenvectors with the highest eigenvalues (*highest eigenvalues corresponds to the most variance along that axis, represented by the direction of the eigenvector*).
These eigenvectors are used to create a new space ($k$-dimensional) where the points will be projected.

The number $k$ is picked relatively to the percentage of information to be preserved, $t$.

Distributed Stocastic Neighbor Embedding preserves local structures, it's capable of handling non-linear structures (if different part of the space behave differently). This approach is useful to visualize 2D and 3D data. 
It transforms distances in probabilities and tries to learn, with a low-dimensional and similar distribution, the high-dimensional distribution. 