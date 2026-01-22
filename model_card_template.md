# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model is a Random Forest Classifier designed to predict whether an individual's income exceeds $50,000 per year based on census data.
The model uses demographic and employment-related features extracted from the 1994 Census database. The model uses the scikit-learn framework.

The model uses parameters for number of estimators=100, a maximum depth of 10, random_state of 42, and n_jobs=-1.

Input categorical features are workclass, education, marital-status, occupation, relationship, race, sex, and native-county.

Input continous features are age, fnlgt, education-num, capital-gain, capital-loss, and hours-per-week.

## Intended Use
This model is built for educational and research purposes, so intended users are data analysts/scientists, ML engineers learning about model deployment.
Researchers studying income prediction or ML scalable pipelines, including students, may also be interested.

This model should NOT be used for any high-stake decision-making without human oversight, including hiring, credit decisioning, or any other production system.


## Training Data
Dataset: UCI Adult Census Income Dataset
Source: Extracted from 1994 Census database
Original size: 32,561 records
Training size: 26,048 recods (80% of original)

Target variable (salary) was used to stratify the data to maintain class distribution after the split.
Binary classification of salary above or below $50k annually.
Data is preprocessed using sklearn's OneHotEncoder.

## Evaluation Data
Testset: 6,513 records (20%)
Selection method: stratified random sampling
Preproccessing uses same fitted encoder and label binarizer as training data.

Slice level analysis on categorical feautures allows idetification of fairness issues between demgoraphic groups.

## Metrics
Metrics:
    Precision: proportion of positive predictions that are correct (TP / (TP+FP))
    Recall: proportion of actual positives are are correctly identified (TP/(TP+FN))
    F1 Score (F-beta with B=1): harmonic mean of precision and recall

Overall perforamnce:
    Precision 0.8058
    Recall: 0.5453
    F1 Score: 0.6504

When the model predicts someone earns over $50k, it is correct 80% of the time.
But the model only correclty identifies 55% of people who earn over $50k.
The f1 score shows a balance leaning in favor of precision.
The model is more conservative in predicting the positive class, meaning it is suitable for situations where false negatives are preferred over false positives.

## Ethical Considerations

Bias
The model is trained on 1994 data, which may not reflect current socioeconomic realities.
Protected classes (race, sex, country of origin) are used as features, which could cause discrimination.

Privacy
Training data comes from public records with no personally identifiable information (PII)

## Caveats and Recommendations
Model Limitations
    Model is trained on 1994 data, which is over 30 years old. 
    $50k adjusted for inflation today would be a lot higher.
    Binary classifcation oversimplifies economic status without consideration for regional cost differences.
    Model may not work well with non-US poulations, current economic conditions, or industries not represented in the dataset.

    This model exist for training purposes only and further testing is still needed.