#  Chatbot-deep-learning

This is a chatbot I created after scraping tons of reddit data and answers.com. The final dataset was 3.5 million lines after extensive filtering to produce clean and clear data. Below is a sample conversation.












How to get started:

  Clone this project to your computer.
  
  I prefer pycharm and anaconda. If you already have anaconda, create a new environment like this:
  
    * conda create -n chatbot tensorflow-gpu==1.15.0
    * pip install -r requirements ( or pip install individually )
    * finally, I use the extremely useful Spacy module for NLP. pip install spacy first, then python spacy           download en_core_web_sm. For additional information, see spacy documentation.



Run my pre-trained model
Download my pre-trained model (3 GB). Place model into 70milmodel folder. All other associated files should already be available.

Run interact.py (preferably in pycharm with ctrl-shft-f10 or ctrl-f5). Wait for it to load up. Start conversating. 

You can play around with some of the variables but after extensive testing I have determined these to be close to ideal.

The beam width by default is 5. I advise you not to alter this unless you want to make further changes. Part of the strategy to make the chatbot sound realistic is generating 5 answers and then engaging in a procedure to choose the best one based on different factors. 

temperature: 

to be contd

Here is a link to the trained model. It is 3Gb. Place into the models/70milmodel folder. Happy chatting. https://www.dropbox.com/s/4gn7lnk9nblkzmc/model-24750.data-00000-of-00001?dl=0

tensorflow==1.15.0


