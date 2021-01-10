# AzureMLEngineerUdacity

## Problem to solve:

The problem is to predict a variable depending on the value of different variables.

# How it has been solved:

The problem has been solved by deploy in azure cloud service the dataset, different virtual machines, and algorithms to find the best hyperparameters.

## Architecture, data  hyperparameters, classification algorithms

The dataset was loaded in the azure dataset but it was loaded in the training script by TabularDatasetFactory and Dataset. Tabular objects. 
The algorithms must predict a categorical variable this means that this machine learning algorithm must be a classification algorithm. The first algorithm was logistic regression.

## The explanation of why choose parameters

The hyperparameter chose is a very crucial point, the reason to train different hyperparameter is to improve the performance of the algorithm selected. There are a lot of different possible combinations of hyperparameters and try all of them would take a long time. The way to do this much faster is to sample hyperparameter from a random distribution. Depending on the hyperparameter value the values can be uniform from a  range of values, choose from a discrete number o values and another continue distribution for example normal distribution. In this example, the max-iter parameter can take a very long range of values then discrete and short values were provided to choose and c can take any value from 0 to 1 then a uniform distribution is generated each time this parameter is required. The way to use this hyperparameter selection is the next: each timestamp that the model is training choose from this random distribution the hyperparameter of that execute; this execution fits the data and computes the score, in this case, accuracy, the next run, if the early stoping policy is not riched, select randomly another value from the hyperparameters distribution

In the first script run the hyperparameters were --C=0.89, --max_iter=10. the reason why these values have been chosen is that it was the first try that gives a good accuracy score. 
the second script, hyperparameter search try to find best c and max iter by select from random values, max_iter was chosen by a discrete number of possibilities choice(10, 50, 70, 100), and c was a uniform form 0.05 to 1

## Explanation policy 

The early stoping policy is very important because it helps to save compute time and reduce overfitting. Every iteration the model is going to improve, if it doesn't improve enough stop the train is the best idea. The way to stop without human supervision is by defining an early topic policy, for example, improve less than epsilon, a small difference between new and previous desire metric.

The policy that has been choosed is BanditPolicy(evaluation_interval=2, slack_factor=0.1) The frequency for applying the policy=2 (1) and 

The ratio used to calculate the allowed distance from the best performing experiment run=0.1 (2)
This means that (2) the new best model must increase by 0.1 the previous one and the policy (1) is executed every 2 timestamps 



## Describe automl model and hyperparameters:

The automl is going to perform multiple models to maximize accuracy. To define this object the next parameters are selected:
- Experiment_timeout_minutes=30 time to iterate automl
- Task = 'classification' problem to solve type, this option selects what type of models are going to be performed.
- Compute_target=compute_target VM where the code is going to be deployed
- Training_data = dataset dataset loaded in memory
- Primary_metric= 'accuracy' metric to maximize or minimize
- Label_column_name = 'y' name of the target column in the input dataset
- n_cross_validations=5 number of k fold that each algorithm is going to perform to ensure to have the best models. the value select was 5, The means that the entire dataset is going to be split into 5 datasets. each iteration the algorithm, in that run, is going to training a hyperparameter by using 4 of 5 split dataset and compute the validation metric with not seen dataset (1/5)




Every iteration an algorithm is going to ver performed, we can't see the hyperparameters of each algorithm but we can see which one has better accuracy, maybe in a farther step agreed search to find to best hyperparameter would improve the scores.

| ITERATION |  PIPELINE                                    |   DURATION   |   METRIC |     BEST   |
| --------- | -------------------------------------------- | ------------ | -------- | ---------- |
|         0 |  MaxAbsScaler LightGBM                       |   0:00:49    |   0.9144 |   0.9144   |
|         1 |  MaxAbsScaler XGBoostClassifier              |   0:01:01    |   0.9149 |   0.9149   |
|         2 |  MaxAbsScaler RandomForest                   |   0:00:46    |   0.8937 |   0.9149   |
|         3 |  MaxAbsScaler RandomForest                   |   0:00:46    |   0.8880 |   0.9149   |
|         4 |  MaxAbsScaler RandomForest                   |   0:00:51    |   0.8083 |   0.9149   |
|         5 |  MaxAbsScaler RandomForest                   |   0:00:46    |   0.8025 |   0.9149   |
|         6 |  SparseNormalizer XGBoostClassifier          |   0:01:19    |   0.9114 |   0.9149   |
|         7 |  MaxAbsScaler GradientBoosting               |   0:00:59    |   0.9042 |   0.9149   |
|         8 |  StandardScalerWrapper RandomForest          |   0:00:48    |   0.9005 |   0.9149   |
|         9 |  MaxAbsScaler LogisticRegression             |   0:00:55    |   0.9085 |   0.9149   |
|        10 |  MaxAbsScaler LightGBM                       |   0:00:53    |   0.8930 |   0.9149   |
|        11 |  SparseNormalizer XGBoostClassifier          |   0:00:59    |   0.9122 |   0.9149   |
|        12 |  MaxAbsScaler ExtremeRandomTrees             |   0:02:21    |   0.8880 |   0.9149   |
|        13 |  StandardScalerWrapper LightGBM              |   0:00:49    |   0.8880 |   0.9149   |
|        14 |  SparseNormalizer XGBoostClassifier          |   0:02:14    |   0.9132 |   0.9149   |
|        15 |  StandardScalerWrapper ExtremeRandomTrees    |   0:01:03    |   0.8880 |   0.9149   |
|        16 |  StandardScalerWrapper LightGBM              |   0:00:56    |   0.8880 |   0.9149   |
|        17 |  StandardScalerWrapper LightGBM              |   0:01:03    |   0.9074 |   0.9149   |
|        18 |  MaxAbsScaler LightGBM                       |   0:01:04    |   0.9048 |   0.9149   |
|        19 |  SparseNormalizer LightGBM                   |   0:01:01    |   0.9139 |   0.9149   |
|        20 |  SparseNormalizer XGBoostClassifier          |   0:00:54    |   0.9119 |   0.9149   |
|        21 |  MaxAbsScaler LightGBM                       |   0:00:54    |   0.9090 |   0.9149   |
|        22 |  MaxAbsScaler LightGBM                       |   0:00:57    |   0.9118 |   0.9149   |
|        23 |   VotingEnsemble                             |   0:01:36    |   0.9171 |   0.9171   |
|        24 |   StackEnsemble                              |   0:01:35    |   0.9150 |   0.9171   |
		
		

## Comparison of their models and their performance

The Automl train a lot of different models; LightGBM, XGBoostClassifier, RandomForest, GradientBoosting, LogisticRegression, ExtremeRandomTrees with different preprocessing steps to find features; MaxAbsScaler SparseNormalizer and two ensemble models voting and stack. Ensemble models combine all the models in only one.
The best algorithm is voting ensemble it is a completely different architecture to logistic regression, the model of the previous section. The voting system gives a score to each model to have a more precise prediction. 
 

## improve the model:

One thing that could improve all this metric is to modify the execution time of the automl to find more models, anther thing is to perform a hyperparameters search with more hyperparameters. Another thing to improve the model is to analyze the error, maybe the class is not balanced, maybe we need to split the data into some subgroups. A descriptive phrase would help in this task.

The reason why is very important to do an EDA(exploratory data analysis) is that if the dataset is not balanced the model will not learn to predict one of the class. There are multiples ways to have balanced classes, one of them is to sample repeated cases in the less represented class.

The reason why is very important to analyze the error is that we can find a pattern that all miss classify cases are following. Maybe a nested model would learn how to predict these miss classify cases.

One of the best model has been selected, a deep search in the hyperparameters selected would improve the accuracy of the model.
