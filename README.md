# wikipedia-to-news-gui
## GUI for Stanford CS224N Final Project 2020
### Kevin Ji, Juliette Love, Moritz Stephan

This is the GUI for out Fact detector. We trained a CNN model with a stacked Highway gate on news articles relating to wikipedia changes. The focus was on changes that were clear changes, e.g. athletes switching teams. The sentences in the articles that contained the fact were labelled using TF-IDF. This model also makes use of Bert embeddings for enhanced results.  

### Instructions to install and run this demo:  

Open the terminal and navigate to the folder where you want to put this project. 

Clone the GitHub repository  <br>
`git clone https://github.com/austrian-code-wizard/wikipedia-to-news-gui`  
Navigate into the directory  
`cd wikipedia-to-news-gui`  <br>
Install the virtualenv tool for python  
`python3 -m pip install virtualenv`<br>
Create a new virtual environment  
`python3 -m virtualenv venv` <br>
Activate the virtual environment  
`. venv/bin/activate` <br>
Install dependencies  
`pip install -r requirements.txt`  <br>
Run the app  
`python app.py`  <br>
Now, open your browser and navigate to the site [http://127.0.0.1:5000/](http://127.0.0.1:5000/)
