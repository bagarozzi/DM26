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

### Feature transformation
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

**Temporal leakage** avoids that data produced *after* the observation is used during training so that nothing is learned from the future, rather the future is used for *validation*.
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

### Feature transformation
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

**Temporal leakage** avoids that data produced *after* the observation is used during training so that nothing is learned from the future, rather the future is used for *validation*.

Temporal feature are **circular**, we must use specific distance equations (e.g. 23:00 to 1:00am). Another way is to use the **sine/cosine distance** where the value is put on a circle and then it can be compared using angular units (rad, deg), given these i can also compute the mean time of some operations.

## Skipped lesson (2/11/26)


## Anomalies
Temporal feature are **circular**, we must use specific distance equations (e.g. 23:00 to 1:00am). Another way is to use the **sine/cosine distance** where the value is put on a circle and then it can be compared using angular units (rad, deg), given these i can also compute the mean time of some operations.

## Skipped lesson (2/11/26)

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

