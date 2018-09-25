# Flight Delay Prediction

[This project](https://hackmd.io/s/SyXikdg_g#Flight-Delay-Prediction) is based on an assesment from [Terminal 1](https://www.terminal1.co/).
The objective is to build a predictive model forecasting flight delay with the final purpose to minimize the gap between claimed and budgetted fee.

## Description of the problem and solution
### High level description
Using a dataset of almost 900,000 past flights, the objective is to build a model which will predict the amount that will be claimed on each flight knowing that flights having more than 3 hours delay or cancelled are subject to a a claim of 800. Other flights end in no claim.
### Approach
To tackle the project, we go through the standard data science process:
  *  Explore the dataset
  *  Enrich (feature engineering)
  *  Model
  *  Solve
### Predictive model
I have built the model in 2 steps:
*  Predict whether a flight will be claimed (binary prediction)
    * trying logistic regression and random forest, optimized with grid search
    * collecting the actual probability associated with the predictions
*  Perform polynomial optimization on each probability knowing the quantity of right / wrong prediction in the training phase
### Detailled explanations
The notebook is fully commented for you to understand the rationale of the project.
Likewise, a powerpoint file summarizes the whole project and results is also available. To read!

## Getting Started

There are 2 ways to use my project:
* Using the jupyter notebook, running the entire notebook and then inputing your own dataset in the very last function to obtain the predictions
```
flight_predictor('your_file.csv')
```
* Going online at 54.210.91.79 to make one predicition using a webservice hosted on AWS uploading your 'your_file.csv'

## Prerequisites

* For **online prediction**
  *  You need... nothing! An Internet connection... at least.
  *  Go to 54.210.91.79
* For the **jupyter notebook**
  *  anaconda 4.5.8
  *  python 3.6.4
  *  numpy 1.14.0
  *  pandas 0.22.0
  *  matplotlib 2.1.2
  *  seaborn 0.8.1
  *  scikit-learn 0.19.1
  *  folium 0.6.0
  *  branca 0.3.0
 If you wish to install a **flask app** on AWS
  *  python 3.5.2
  *  flask 1.0.2
  *  numpy 1.15.2
  *  pandas 0.23.4
  *  scikit-learn 0.19.2
  *  html5lib

## Installing

Download and install [anaconda](https://www.anaconda.com/) and the libraries mentionned above.
Example to install flask:
```
pip install flask
```
Put the files in a folder on your machine, start the notebook and you are set.

You may also need an [AWS account](https://aws.amazon.com/)
* Here is a wonderful [tutorial](https://ketakirk.wordpress.com/deploy-an-app-on-aws/) taking you from account creation to flask deployment. Brilliant
* Choose the free tier vm VM Ubuntu-xenial-16.04-amd64-server
* All the files are available to you EXCEPT the key to access the machine. You would create your own

## Running the tests

The tests are at the end of the notebook. They consist in uploading a series of files to test:
* the model itself with valid files
* the robustness of the solution with unvalid files
```
flight_predictor('1_flights_sample_valid.csv')
```

## Deployment - additional notes

Make sure to place your file on the AWS VM into a folder names "flights" so that the absolute path is '/home/ubuntu/flights'. If not, change the file 'principal.py' with the new route.
If you want to change the online predictive model, build the updated model with jupyter, save the model as 'model.pkl' and replace the existing file in the 'external_data' folder in the VM.

## Built With

* [Jupyter](https://jupyter.org/) - Notebok
* [Flask](http://flask.pocoo.org/) - Web framework
* [Atom](https://atom.io/) - Editor
* [AWS](https://aws.amazon.com) - Cloud solution

## Contributing

You are welcome to add a CONTRIBUTING.md file should you want to join.

## Versioning

We use Git for versioning.

## Going further

There are multiple areas of improvement for us to explore:
* Addition of new external data
  * Weather forecast… but not free!
  * Add bank holidays of destinations
* Model
  * Investigate further resampling to balance claimed and not claimed
  * Introduce time series to capture changes over time
* Web app
  * Enhance architecture: more powerful instance, auto-scaling, load balancer and database to store the predictions mode
  * Provide json (API-style) and csv file and live result for 1 choice

## Authors

* **Romain Barraud** - *Initial work* - [LinkedIn profile](https://www.linkedin.com/in/romain-barraud-6722694/)

## License

This project is free of use.

## Acknowledgments

This project and most of the knowledge I have humbly acquired in the fascinating field of data science is in a way or another bound to the [Coursera](https://www.coursera.org) and [edX](https://www.edx.org), the 2 best MOOC platforms where outstanding data science, AI and big data courses can be found and enjoyed.

