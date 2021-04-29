In order to run the code locally, you will need to setup a Conda environment.  You may need to setup the Anaconda to the PATH on your computer first.  The environment we used is stored in the 'yelpers_project.yml' file.  In order to install this environment, open a command line in the same directory as the yml file and run the following command:

conda env create -f yelpers_project.yml

Once installed, activate the Conda environment by running this command:

activate yelpers-project

We highly recommend running our code on Google Colab.  Load up any 'yelp_humor_classification_bert.ipynb' or its derivatives on Google Colab.  Also, load up the .csv files we stored in the 'data' folder in Google Colab.  We high recommend to run the notebook with a GPU setup.
