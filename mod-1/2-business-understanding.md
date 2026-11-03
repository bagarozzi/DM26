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

## Skipped lesson (20/10/26)

## Sampling techniques

Differnet sampling techniques are adopted in different phases of the process: 
- During the observation of a phenomenon it's better to have a dataset that represents very well all the population
- During the training for a specific task of a ML model it's better to have a dataset that fits with the training and with
the task of the model, that's not necessary representative as much as the ML model.

The evaluation dataset has to verify generalisation so it has to be as close to the population as possible.

While **collecting a population sample** for the observation phase it's important to:
- the population, as a whole
- the target population, the actual cases on which the model will be deployed
- sampling frame: the examples that will be analyzed

the gap between the target population and the sampling frame is the main source of *selection bias*:
- there could me individuals that are more present or partecipative -> *self-selection bias*
- only the surviving individuals are included in the frame -> *survivorship bias* 
- some data reflects past conditions or times that no longer hold during the observation, learning and inference phase -> *historical bias*

Sampling has to be done randomly and in different rounds because of variability and instability. **Stratified sampling**
is useful in presence of rare classes, here the dataset is partitioned in classes by an attribute and a percentage is taken from each class, so that the sampling is *proportional*.
This provides control over the representation of a subgroup (the one used for partitioning) but does not guarantee the same for the other subgroups.

In **random oversampling** elements are duplicated from a minority class to increase its *weight* so that an optimal distirbution is achieved. This ***increases overfitting*** because it does not learn the general pattern but rather reinforces the presence of specific minority cases. 

**SMOTE** (Synthetic Minority Oversampling Technique) is another way to manage minorities. It produces synthetic samples by **interpolating in between neighboor existing** samples. 

*Where you generate matters more than how much*, SMOTE-2 generates around the decision boundaries.

### Measuring performance with oversampling
After oversampling for the training population, during the validation phase, normal accuracy calculation may be misleading due to the unbalanced population: *the classfication errors on the minorities are **averaged out** by the successes of the majority*.
**Precision** and **Recall** measure respectively false positives and false negatives, it works better with class-specific metrics.

**Oversampling improves recall**, improves classification of minorities so there are less false negatives. Precision may increase or remain the same. 

### Cost-Based Evaluation
While measuring positive classification, Precision and Recall assume that false positive and false negative have the same weight at inference time. But it's not always like this: in some instances it's better to have more false negatives than false positives. 
In this case we want to *minimize false negatives*. 

Cost-based classifiers are trained using a **cost function** that tells the system the *cost* of an error. This changes the main cost function and forces the classifier towards producing less false negatives than false positives, possibly at the spite of some accuracy.

## Dataset shift 
When the dataset on which the model was supposed to work, while in production, **changes**.

- **Covariate shift**: the distribution of the features is different. The system is still capable to recognize, but the input is different.
- **Label shift**: features remain so the system is still capable to classify but the distribution of the classes is different, so it's a matter of calibration.
- **Concept drift**: the underlying rules are changed (for example, economical rules learned by a model have changed). The only thing to do here is retrain. Concept drift can happen in different ways: sudden, gradual, linear/incremental, recurring.

**Detecting drift**: it depends on if I am aware of the label changing (*is this being classified as another class?*) or if there's no label (*some model results have a delayed effect or result*). We can monitor accuracy, F1 or precision on a sliding window; or by using the **Drift Detection Method** by comparing the expected probability against the actual probability.

**Population Stability Index** measures how much the distribution of a variable has changed betwene the reference population and the current population. 

## Skipped lesson (26 and 27/10/26)