************
INSTRUCTIONS
************

This code base include in my project submission is a refined demo illustrating the pinacle
of all work carried out in my project, described in detail in the report.

The required Python libraries to run the demo are listed in 'requirements.txt' and can all
be installed using Python's pip module (https://pip.pypa.io/en/stable/installing/)

Note: Depending on your machine, and what other libraries you currently have installed,
you may be required to download extra libraries which will be specified when you try to
run the code.

This demo works with Apple's daily data for the last several years.
Unfortunately, I have not included a demo of the intraday data with sentiment analysis,
as my API authorisation to retrieve the intraday market prices expired this week.
(Anyway, as explained in the report, positive intraday results are yet to be achieved.)

To run the demo, a command with the following format must be executed in the root directory:

    python Main.py -generate yes -restore no -train yes -simulate yes
    
    or condensed as
    
    python Main.py -g y -r n -t y -s y
    
All arguments must be either specified as y/n.

- generate will retrieve, build and restore the data set in the Data/ directory.
  (this must be run at least once prior to running the demo

- restore will load any previosuly trained model stored in the SavedModels/ directory.
  (I have included a trained model which will be loaded if this argument is set to 'yes'.)

- train will perform offline training on a model on the training set for 250 epochs.
  (this will take some time - the number of epochs can be reduced in Constants.py)
  
- simulate will run a simulation of the trained model on the last 100 days of the data set
  and will plot out the prices for these 100 days, as well as the ROI that the model would
  achieve by executing trades based on its predictions.

************
REQUIREMENTS
************

numpy==1.14.0

tensorflow_gpu==1.5.0

pandas==0.22.0

TA_Lib==0.4.16

Quandl==3.3.0

alpha_vantage==1.8.0

seaborn==0.8.1

matplotlib==2.1.1

tensorflow==1.7.0

