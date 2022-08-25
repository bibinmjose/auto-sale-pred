## Solution Design

 - Model will be used to predict future car value. Since the model will be used for furture car value predictions, train/test split should be done on a date (Sold date).
 - Model serving requires running an inference pipeline every day since some features like age is calculated in days
 - 

## Solution Approach
 - KNN imputer
 - Feature Engineering
 - Scaling & Encoding

## Model Training and Testing

#TODO: Train - Test split date

### R-2 Score comparison
| Model            | Train | Test  |
|:-----------------|:-----:|:-----:|
| Ridge Regression | 0.785 | 0.679 |
| XG-Boost         | 0.953 | 0.837 |

![Model_Result](model_result_comp.png)

## How to Reproduce