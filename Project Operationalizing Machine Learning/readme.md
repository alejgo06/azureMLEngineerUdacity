# An overview of the project
In this project, the main objective is to deploy machine learning models in azure. To do this there are a few things 
that are required to do:
- Get permission
- Create computer cluster and load dataset
- Train a model
- Deploy model
- check logs and swagger
- Predict new data, entry point, and benchmark
- Publish pipeline

# An architectural diagram
The structure of this project can be shown as this diagram:
Each box is a step in the project and each box has multiples tasks to do to be completed.

![Screenshot](img/diagram.png)

# A short description of how to improve the project in the future
To improve this project a few things must be made:
- Understand the data. this means to do EDA to understand the models. Azure automl is looking for multiple models,
 multiples hyperparameters of each model and some preprocess steps but it is known that the vast majority of the job of a data scientist is feature engineer. An EDA should help to get a better comprehension of this problem to be able to 
 build new variables, then get better predictions(more than 91% accuracy)
 
- This model uses the only accuracy but maybe the problem requires giving more weight to one of the classes. In that case an
 F beta score should be a better option.
 
- Train deep learning models. In this case, I uncheck deep learning models, the reason why is because 3 hours, 
the time that I have available the Udacity lab, is not enough time to train too many models so I decided to skip deep learning models that should take a longer time. Deep learning models can be more complicated to explain to people with fewer analytics skills than another model such as random forest but It can take a very high accuracy.

- In this project, I learn how to deploy ml model in azure and consume it form an entrypoint and jupyter notebook but It 
could be grateful to use it in an app for example a Django, dash Streamlit app, or a simple HTML website.

- Another important point could be to deploy the model in a docker microservice and deploy this model with azure
 Kubernetes stances, the reason why to add this section is that it would teach how to deploy our own docker

# All the screenshot required in the project main steps with a short description 
In this section, you can see the step by step entire project. The project was made in two days and I had to repeat from
 the beginning but I skip to upload repeated pictures.
* Authentication
![Screenshot](img/Captura.PNG)
 
* create compute cluster
    ![Screenshot](img/Captura8.PNG)
* Automated ML Experiment
    * Dataset
    ![Screenshot](img/Captura2.PNG)
    * automl configuration
    ![Screenshot](img/Captura3.PNG)
    ![Screenshot](img/Captura9.PNG)
    * model experiment complited
    ![Screenshot](img/Captura11.PNG)
* Deploy the Best Model
    * select best model
    ![Screenshot](img/Captura6.PNG)
    * deploy configuration enable authentication and ACI
    ![Screenshot](img/Captura12.PNG)
    * deploy
    ![Screenshot](img/Captura13.PNG)
* logs 
![Screenshot](img/Captura14.PNG)
![Screenshot](img/Captura15.PNG)
* Swagger
![Screenshot](img/Captura21.PNG)
![Screenshot](img/Captura22.PNG)
![Screenshot](img/Captura23.PNG)
* Consume Model Endpoints
    * endpoint
    ![Screenshot](img/Captura17.PNG)
    ![Screenshot](img/Captura18.PNG)
    ![Screenshot](img/Captura19.PNG)
    ![Screenshot](img/Captura20.PNG)
    * benchmark
    ![Screenshot](img/Captura24.PNG)
    ![Screenshot](img/Captura25.PNG)
* Create, Publish and Consume a Pipeline
![Screenshot](img/Captura26.PNG)
![Screenshot](img/Captura27.PNG)
![Screenshot](img/Captura28.PNG)
![Screenshot](img/Captura29.PNG)

# A link to the screencast video on youtube.
In the next link, you can see a demo of this project:  https://youtu.be/lkBe-09wGPE

[![IMAGE ALT TEXT HERE](https://bcs.solutions/wp-content/uploads/2019/04/Azure.png)](https://youtu.be/lkBe-09wGPE)