Principles: in DCAI we put focus also on data analysis and not only on model fitting and training.
- Improvement on **data fit**
- Improvement of **data consistency**: removing noise, inconsistencies, correlation
- **Improvement throught iteration**
- **Human-centric**: transparency, ethical awareness and control over the process
- **Sociotechnical**: It's not just model optimization, it supports human agency and augment human capabilities and so it has to be used responsibly.

**Crisp methodology**: a structured and iterative method on how to conduct a data mining project.
It's centered around data, in the first part it's constructed a basic understanding of both the business context and the data provided, then data is prepared appropriately and the model is chosen. In the end there's evaluation and deployment.

> *Diminishing results in ML:* typically we can get a better result at the beginning of the data analysis project but at some point the returns get lower.

When should i stop improving my model? Decide a **definition of done** for the *domain problem*. In this way we stop as
the valude produced by the project reaches optimal result before the returns get lower.
In this context there are *many good solutions*, the hard part is not getting lost in tradeoffs.

Attributes similarity: seen that zero-values often not make sense, when we have to compare the similarity of attributes that
are nullable we have only to compare the non-null values and ignore the others (if we compare the null values and have the majority of them then two data points will be similar).

