
Two main types:
- **Close-ended**: limited number of outputs and a few correct (*sentiment analysis, named entity recognition, multiple choice*)
- **Open-ended**: 

Human-based evaluation uses humans:
- Likert: assign an absolute score to each outputs, two annotators' votes may not be comparable.
- Preference: compare multiple outputs and assign best/worst, add a *None* label and hide any system identifier.

Clearly, instances have to be sampled in a statistically-relevant way (reproducible, big...), generalizable and variable. 

IAA, Inter-Annotator Agreement. How much annotators agree wth each other beyond chance? Measuring how much they agree and how much they should agree *randomly*.
Different metrics available: Cohen, Weighted Cohen, Fleiss, Krippendorff.