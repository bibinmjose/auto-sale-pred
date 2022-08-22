## Solution Design

 - Model will be used to predict future car value. Since the model will be used for furture car value predictions, train/test split should be done on a date (Sold date).
 - Model serving requires running an inference pipeline every day since some features like age is calculated in days
 - 