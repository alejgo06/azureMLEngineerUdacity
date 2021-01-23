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
Audio transcription:
- 0:13 I need to have some extra time for my lab
- 0:22 Now, I can come back to the lab
- 0:24 I show you the uploaded dataset.
- 0:27 Is this dataset
- 0:32 Next, I show the automl ml model that I have trained.
- 0:34 It is this one. 
- 0:35 It has been completed in 1h and 22 minutes.
- 0:40 let's explore this automl run
- 0:42 It's status is completed. 
- 0:42 It gets 91.83 accuracy
- 0:47 the best model is a voting ensemble
- 0:51 let's see the configuration of this run
- 0:54 accuracy as the primary metric, without deep learning models to save time. 
  On hour train time and 3 max concurrent iterations (previously, I had created a VM with 4 cores)
- 1:05 let's see the models
- 1:06 this is the voting ensemble, the model that I have deployed.
- 1:10 we can see the endpoint
- 1:13 this is the endpoint
- 1:16 let's see the configuration of this endpoint
- 1:19 this is the rest endpoint that I need for de endpoint script, is the URL where the model is hosted, where I will send an HTTP request
- 1:22 this is the swagger URI. It is the JSON that I have upload to the swagger docker to see the documentation of the API of this deployed model.
- 1:31 let's check the logs of this endpoint. To do this I have copied and past logs.py in this jupyter notebook.
- 1:34 this is the same code that you can see in the logs.py and you can see that I have to change the name of the deployed 
- 1:36 it is the same name because I want to see the logs of this deployed model
- 1:40 I am sorry this is not the notebook
- 1:41 this is the notebook
- 1:44 when I click run we can see the output logs
- 1:52 The next thing is to predict data, use the endpoint. I have copied to code from endpoint.py and pasted it in this notebook.
- 1:56 but first look at the swagger. It is very friendly documentation of this endpoint.
- 2:00 we can see that this swagger is associated with my endpoint because at the top of this documentation you can see deploy-model as the title. the same name of the deployed run.
- 2:12 I am looking for the scripts that I had to modify to create this swagger documentation. 
- 2:15 I have downloaded the JSON
- 2:16 let's see the script. int his notebook the .json provided is going to be deployed in port 8000 to be visible to the swagger docker
- 2:21 but this is the important file, the .sh 
- 2:24 in this .sh the swagger docker in the run-in port 9811 and refer to the internal port 8000 to communicate the information from the previous swagger.py that I saw you before.
- 2:28 a this is the swagger.json
- 2:34 As I have mentioned port 9011 in the .sh 
- 2:37 localhost port 9011 in my browser
- 2:39 the docker shows this information
- 2:45 we can see the documentation, the healthy response, how to curl request is build
- 2:50 how to data input should look like
- 2:54 we can click run and execute 
- 3:27 let's come back to the notebook
- 3:29 this is new data that I am going to predict. All this code come from the endpoint.py script
- 3:32 this is the data
- 3:35 I click run
- 3:36 and the output results "no"
- 3:39 we can see that that input data have been saved in a JSON to be readable form the API rest
- 3:40 this is the data.json
- 3:45 next, let's run a benchmark of this endpoint. To do this we need to open a terminal.
- 3:55 this is the terminal
- 4:18 I am going to open a previous terminal session
- 4:21 terminal 2 is the terminal that I have used previously 
- 4:30 and I run the command to benchmark this endpoint with the library called ab.
- 4:32 The benchmark is running
- 4:33 this is the performance 
- 4:41 the information size is 240 bytes
- 4:43 0 errors from 10 request
- 4:47 the time was 1.20 seconds, it is very fast.
- 4:52 more information about the performance if we want more details 
- 4:56 then let's see the deploy pipeline notebook
- 5:00 It is running right now
- 5:11  I have run all the cells, I have to change the compute cluster and experiment name
- 5:14 and we can see the intermediate output of the training of the automl
- 5:27 lastly let's see the pipeline in the UI and how it is running
- 5:33 this is the pipeline
- 5:35 it is running
- 5:36 let's see the run
- 5:38 in this graph we can see that this is a very simple pipeline take the data and automl model
- 5:39 this is the end of this video. Goodbye!





[![IMAGE ALT TEXT HERE](https://bcs.solutions/wp-content/uploads/2019/04/Azure.png)](https://youtu.be/lkBe-09wGPE)
