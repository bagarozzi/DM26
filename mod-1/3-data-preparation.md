# Data preparation
Activities such as: encoding, cleaning, completion, resampling.

## Encoding techniques
It's better to work with numbers, so categorical data has to be encoded (transformed) into numbers so that it
can be used with ML models. 
**Embedding** is an encoding process that maintains meaningful information about a feature, so that *similar items are close to each other in the vector space*. Embedding captures information into hidden, compact data that is not directly linked to what it's representing.

>*PCA* can be considered a embedding technique.

- **One-hot** encoding represents each label/category throught binary attributes. If there are 5 categories, then there are 5 binary labels, activated one at a time.
- In **Ordinal encoding** each category is mapped into an integer reflecting the natural order of categories (*low, medium, high*) -> this method distorts distance measurements
> To *encode* ordinal attributes we can use the feature as a *target variable* (continuous, ordinal value) to map it to a category
- **Target encoding** maps features' values (target variable) to a category. A category's value may be the *mean* of each target variable.

> First, samples are divided in categories and its computed the mean of each category's target variable. Then, if some categories violate the monontonic ordering, the **isotonic regression** is computed: the two categories are merged and the encoding is replaced with their weighted average.

## Data cleaning
- Improve **accuracy** by *representing real-world value precisely*. 
- Ensure that data is **complete**
- Ensure that data is **consistent**
- **Uniqueness**: no duplicates in the dataset
- **Timeliness**: updated data
- **Semantic validity**: data corresponds to the rules of the domain

### On missing data
Loss of statistical power, biases, can't work with some ML models.

Data can be missing: at random (*a sensor fails*), depending on some other feature (causality), depending on the values itself.

How to handle missing values:
- **Delete** the point with a missing attribute
- **Replace** with "unknown"
- Substitute the missing value with some **statistically-retrieved value** (*average may underestimate the variance and introduce bias*), replace with the most recent present value (*assumption*) or throught regression
- Use **ML** to fill the missing values (forecasting to forecast)

> **Side effects**: statistical methods *reduce variance* (more points close to the mean), correlations are distorted (less points with correlating attributes), ignoring the problem (missing attributes may be interesting to study by itself).

This approach has to be *integrated in the workflow*: understand and study the missing points, choose and appropriate method to solve the problem, monitor the solution. When training a *model* we have to be careful to **avoid data leakage**. 

## Scaling
If the model involves calculating distances and the values vary widely the calculations will not work properly. If the distance between some features is calculated, the features that vary the most will govern the decision more than others.
KNN, SVM, PCA, K-Means are very sensitive to scaling problems.

The **MinMax** scaler adapts the minimum and the maximum to the $[0,1]$ interval. It's obviously sensible to outliers.

**Robust** scaler uses the interqartile range (IQR) to normalize the values, less sensible to outliers.

**Z-score scaler** maps the value relative to its distance from the mean, divided by the standard deviations. It's sensitive to outliers the same way as its coefficients.

## Sampling techniques

Differnet sampling techniques are adopted in different phases of the process: 
- During the observation of a phenomenon it's better to have a dataset that represents very well all the population
- During the training for a specific task of a ML model it's better to have a dataset that fits with the training and with
the task of the model, that's not necessary representative as much as the ML model.

The evaluation dataset has to verify generalisation so it has to be as close to the population as possible.

While ***collecting a population sample*** for the observation phase it's important to:
- the **population**, as a whole
- the **target population**, the actual cases on which the model will be deployed
- **sampling frame**: the examples that will be analyzed

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

## 26/10/26: Lab 2: sampling techniques