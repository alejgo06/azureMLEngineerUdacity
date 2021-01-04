# azureMLEngineerUdacity
problem to solve:

the problem is to predict a variable depending on the value of different variables.

how it has been solved:

The problem has been solved by deploy in azure cloud service the dataset, different virtual machines, and algorithms to find the best hyperparameters.

architecture, data  hyperparameters, classification algorithms

the dataset was loaded in the azure dataset but it was loaded in the training script by TabularDatasetFactory and Dataset. Tabular objects. 
The algorithms must predict a categorical variable this means that this machine learning algorithm must be a classification algorithm. The first algorithm was logistic regression.

the explanation of why choose parameters

In the first script run the hyperparameters were --C=0.89, --max_iter=10. the reason why these values have been chosen is that it was the first try that gives a good accuracy score. 
the second script, hyperparameter search try to find best c and max iter by select from random values, max_iter was chosen by a discrete number of possibilities choice(10, 50, 70, 100), and c was a uniform form 0.05 to 1

explanation policy 

the policy that has been choosed is BanditPolicy(evaluation_interval=2, slack_factor=0.1) The frequency for applying the policy=2 (1) and 

The ratio used to calculate the allowed distance from the best performing experiment run=0.1 (2)
this means that (2) the new best model must increase by 0.1 the previous one and the policy (1) is executed every 2 timestamps 

describe automl model and hyperparameters:

- experiment_timeout_minutes=30 time to iterate automl
- task = 'classification' problem to solve type, this option select what type of models are going to be performed.
- compute_target=compute_target VM where the code is going to be deployed
- training_data = dataset dataset loaded in memory
- primary_metric= 'accuracy' metric to maximize or minimize
- label_column_name = 'y' name of the target column in the input dataset
- n_cross_validations=5 number of k fold that each algorithm is going to perform to ensure to have the best models. the value select was 5, The means that the entire dataset is going to be split into 5 datasets. each iteration the algorithm, in that run, is going to training a hyperparameter by using 4 of 5 split dataset and compute the validation metric with not seen dataset (1/5)

comparison their models and their performance

the best model was VotingEnsemble




