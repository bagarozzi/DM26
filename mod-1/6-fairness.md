# Fairness
Fairness in machine learning: the set of approaches aimed at detecting and **correcting algorithmic bias** in automated decision processes based on ML models. e.g *absence of discrimination*.
Either it encodes existing human biases or creates new one on the base of statistics.

It's hard to detect because the **accuracy is higher on a biased dataset** and removing protected features does not make it better, that's because the model can *infer information from proxy variables*.

In general, generalisation can be harmful:
- statistics *may be wrong* (arrests are not crimes)
- innocent samples *pay the price for their groups actions*
That's why generalisation are good ***on average*** and can still be incorrect.

Fairness throught unawareness is not always the good case: can create more unfairness, hidden attributes can be useful for accuracy.

Fairness figures have to be introduced to measure fairness over the different tasks in the ML pipeline: for example it's compared the fairness score of the model over different groups with a varying **sensitive attribute**.

During preprocessing: **data reweighting**. Each data point receives a weight that makes the attributes independent from the sensitive ones. Rarer combinations get an higher weight than diffused combinations.

During processing: **Fairness-aware learning**. Optimisation is constrained with a *fairness regularizer* and some tradeoff is made between fairness and accuracy.

During post-processing: **adjusting the decision**. The output is black-boxed, thus its result is adjusted. 

In reccomenders there's strict and complex idea of fairness: recomending a product is not just marketing, it influences people, markets, money and in the end life. Plus, in this systems emerges different type of fairness for each stakeholder.

