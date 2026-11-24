# Features (27/10/26) 
Feature selection: choosing which features to include in the dataset. The choice is based on maximizing performance and generalization for the model.  

We want to remove:
- irrelevant features: provides no information to the cause
- redundant: contains information already contained in other feature(s)

Redundant features can also be harmful to the model, distorting distances due to them weighting more in the space.

## Feature selection algorithms
- filtering algorithm (model-independent)
- wrapper (model-based): choose features that actually improve the model
- embedded (during training)

### Filtering
In **Univariate method filtering**: Apply a statistical scoring function and then apply a threshold. Depending from the target different algorithms can be used: *pearson correlation, ANOVA F-test...

**Multivariate methods** consider the feature itself and its relationship with other features. Checks both relevance and redundancy.
mRMR tries to both maximize relevance and minimize redundancy by using the *Mutual Information* between couples of features.

### Wrappers (model-based)
The solution found is related with the model (optimal). The best features are the ones that maximize the cross-validation score in the subset, here come the subset search methods: greedy, stochastic, limited width.

**Sequential Forward Selection** (SFS) is a greedy algorithm. At each iteration, the feature that adds the greatest improvement is added to the set.

**Sequential Backward Elimination** (SBE): start from a fullset of features and removes the worst features.

Both algorithms stop at a fixed number or when there's not enough improvement anymore.

**Best-First Search**: keeps multiple candidate subsets at the same time. It starts from a feature and branches, choosing multiple other features for the subsets. Typically stop when there's no improvement.

### Embedded
A **Tree-based** approach is useful because only the features that contribute to the fitting of the tree are used. These features are the ones that contribute the most to the *purity* of the tree.
This method is already integrated within the model, while training.

> Logistic regression is a linear classifier that, instead of predicting, computes the probability of an item to belong to the class

**Lasso** models perform both parameter estimation and feaure selection. For this model, adding a feature can be done only if adding it trades off with the penalty added by its coefficient.

> The **XOR interaction** is a particular problem in feature selection: the algorithms that select only one feature at a time fail to detect two features that together create significant information for the model. <br> *SFS, mRMR, Pearson and Univariate suffer from XOR.*

An important thing to note is that: any process that *learns* something from the features must be included in the traning part. If the test samples are selected after the features selection then also them are influenced.

## Feature importance
It quantifies how relevant or influential a feature is. It can support feature selection but *a rule as to be introduced*. 

Decision trees have an intrinsic measure for importance. Decision trees choose the best feature based on the capability of the feature to split the samples on a node.
The importance of a feature comes from the gain of purity to the tree.

F.I. is model-specific but there are also agnostic metrics: **Permutation Feature Importance** computes the importance by making irrelevant a feature and comparing with the original dataset, obtaining the gain (or the loss) of relevance.

# Feature engineering
Improve representation by adding knowledge or expressing data in a way that's easier to learn. 
- An example of the first type might be to compute some ratio or index, such as the BMI. 
- Another way to express data differetly is by introducing a new variable that represents a non-linear relationship between other variables so that the model can learn and infer better

When choosing a model we have to consider the **inductive bias**, when a learning algorithm is chosen then some assumptions have to be made regarding the learning and inference behavior:
- When choosing a linear regressor then we assume that the target function behaves somewhat linearly
- When choosing a k-NN classificator...

- **Feature transformation**: operates on a single feature
- **Feature concstruction**: combination of multiple features
- **Learned representation**: features are learned from the algorithm (PCA actually *learns* a feature and applies it)

## Feature transformation
Feature transformation *operates on the representation of some feature*, these are operations such as: normalization, standardization, scaling, encoding. It changes the distribution, scale or relationship *without creating new information*.
An example is ***distributional transformation***: changes the distribution when the values are skewed or span orders of magnitude. 

## Feature construction
New information that comes from interactions between some features.
- Ratios, differences and products
- Trend and seasonality indicators
- Graph degree and centrality measures

### Ratio
However, ratios can create **misleading correlations**: dividing incidents by number of citizens makes possible to compare cities but not to actually compare the incidence over exposure to risk. Maybe dividing by kilometers travelled in a city is more meaningful and less prone to errros.

Another example is income and debt. The ratio between income and debt capture the *relative* relation between features but misses the magnitude. *Two ratios might be identical while describing two completely **different situations***, the debt/income ratio does not consider the marginal value (captured by residual income), given that *someone who earns a lot of money can still live better after repaying the debt*.

> It's always better to have several small, simple indicators than to have one difficult

### Time-based features 
- Identify the meaning of each timestamps: distinguish between **event time** (when something happens) and **transaction time** when the event is registered by the system
- Pay attention on what is available at inference-time: when anaylising data, keep in mind that only some data produced before the inference was available (*predicted time vs. actual time*).

Time-based features can be extracted from TSDB to understand trending and seasonality of the samples; not only the current state.

**Temporal leakage** avoids that data produced *after* the observation is used during training so that nothing is learned from the future, rather the future is used for *validation*.

Temporal feature are **circular**, we must use specific distance equations (e.g. 23:00 to 1:00am). Another way is to use the **sine/cosine distance** where the value is put on a circle and then it can be compared using angular units (rad, deg), given these i can also compute the mean time of some operations.

TIme series can represent: level, trend, seasonality, irregular variation.
The point of a time series is to turn it into a *snapshot features*; by characterizing what's true today based on the past.

To produce **snapshot features** different operations can be done:
- using *time windows* to detect treding and rates in some periods
- aggregating throught math ops *average*, *sum*, *rates* (*derivatives*), *moving average*
- computing regression to extract a trend

### Structural and relational features
Structural and relational features can be extracted from relationships contained inside samples. This gives a better understanding of degrees of separation, closeness and relations.

In many domains, profile do not come simply as singletons but as a network of interconnected samples. Graphs are the best way to represent and analyse them. 

Some interesting features to extract are:
- Scale
- Cost
- Redundancy

## Anomalies
Causes of anomalies are usually:
- Data from different classes
- Natural variations: a different natural phenomenon can always happen 
- Measurement errors

Some anomaly detection requires **context**, the value is admitted but not in a specific time or place. Some other may be sequences or sets of related data points. 

Some approaches to detect anomalies:
- supervised: when labels are available for both classes, a classification task
- semi-supervised: no labels for anomalies, then data available is deemed "normal" and everything else is an anomaly
- unsupervised

An **anomaly score** or classification is produced. Potential errors of these algorithms may be false negatives and false positive both during inference and during **labeling**.

A **statistical-based** model for detecting anomalies is based over the assumption of a known distribution of data, on which the anomalies are detected. Points *far away* from the mean (based on some thresholds).

...

The problem with some techniques is the variability in density and distance of outliers (what's an outlier in some places it's not in others). **Local outlier factor** (LOF) uses a ratio to determine if the density of a point is low w.r.t. its neighbor.

