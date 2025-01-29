Sentiment Analysis Pipeline

This project is a sentiment analysis pipeline built using the IMDB movie reviews dataset. It gives you the power to analyze text sentiment using two different models:

 Logistic Regression – A fast and lightweight model.
 DistilBERT – A deep learning model for better accuracy.

With a Flask web app, you can enter a review and see whether it's Positive or Negative 

Setup Instructions

1️Install Dependencies

Before anything, install the required packages:
   pip install -r requirements.txt

   
2. Run the data_ingestion.py 
The IMDB dataset will automatically download during data preprocessing, so there's no need for manual setup of the dataset.


3. Run model_training.py
Here we train the logistic regression and dilstilbert, i commented the traning part of dilstilbert because its too time consuming.
The trained model is stored in google drive and below is the link, 

https://drive.google.com/uc?export=download&id=1yy34jTkihUovfSjQQY8ObPQXMCEWhQdF

Remove the comments from the code and run the file and train if has access to GPU or else downlaod the model using the link and move into the Dilstilbert_model folder 
4. Run the app.py
The code finds the logistic regression and dilstilbert model   Once the Flask server is running, http://127.0.0.1:10000
open link and , There enter the text and choose any model between logestic regression and dilstibert and enter submit to see the prediction either its positive or negative 
