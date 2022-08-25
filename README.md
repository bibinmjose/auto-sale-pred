## Solution Design

 - Model will be used to predict future car value. Since the model will be used for furture car value predictions, train/test split should be done on a date (Sold date).
 - Model serving requires running an inference pipeline every day since some features like age is calculated in days
 - 

## Solution Approach
 - KNN imputer
 - Feature Engineering
 - Scaling & Encoding

## Model Results

#TODO: Train - Test split date

### R-2 Score comparison
| Model     | Train | Test  |
|:----------|:-----:|:-----:|
| Ridge     | 0.785 | 0.679 |
| XG-Boost  | 0.953 | 0.837 |

![Model_Result](nbs/model_result_comp.png)

### Explain Model Features
 - Show shap value and PDP plots
 - 

## How to Reproduce

1. `cd` into _root folder_ and follow below
2. `pip install -r requirements.txt`

3. move data to `data/saleprice_dataset.csv`

4. run `dataprocess.py` to generte test/train datset

5. run `embed.py` to generate embedding in `/data/embeddings.pkl`

6. run `train.py` to generate `model.json`