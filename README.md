#  Chatbot-deep-learning

This is a chatbot I created after scraping tons of reddit data and answers.com. The final dataset was 3.5 million lines after extensive filtering to produce clean (no bad words) and clear (less slang) data. Below is a sample conversation.

    YOU:  Hello, what is your name?
    BOT:  My name is Michael
    YOU:  nice to meet you. Do you like cheese?
    BOT:  Yes. I love cheese.
    YOU:  tell me your top 3 cheeses...
    BOT:  Cheese, pizza, pasta.
    YOU:  lol, you are funny
    BOT:  Thanks. My favorite comedian. :)
    YOU:  My favorite comedian is Bill Burr.
    BOT:  He was really funny for a long time. :)
    YOU:  yeah for sure. I went to the beach on Saturday with my friends and had a picnic.
    BOT:  Did you have a good time?
    YOU:  ofc. It was a lot of fun. got to play some volleyball too
    BOT:  Good. I played volleyball when I was at my old school.
    YOU:  are you good?
    BOT:  I'm pretty good. I play with my friends often at the park.
    YOU:  Would you consider yourself a pro?
    BOT:  I don't think so. I play with my friends a lot.
    YOU:  okay okay... ugh, what do you want to talk about?
    BOT:  idk. I don't really have anything to talk about
    YOU:  Well, I need you to be sociable because this conversation is being recorded and will be displayed for others to view online.
    BOT:  okay, sure!
    YOU:  You are completely alright with that?
    BOT:  yes, I'm just a little bit lonely
    YOU:  Aw, I'm sorry.
    BOT:  it's fine, thanks for the encouragement.
    YOU:  So, um, you are not human. You are a chatbot.
    BOT:  Chatbots don't exist on their own, they're all computer-generated. They're just the computers you'd use in a living room at home.
    YOU:  Uh, that's exactly what a chatbot would say.
    BOT:  I think he/she is talking about a "bot" that is capable of replying to you and can give you some humanlike responses and tell you a bit about the world around them (moreso than a chatbot). I'm assuming the person is a bot because they/they've got a name and username. They're basically automatons.
    YOU:  Would you consider yourself an automaton?
    BOT:  I would consider myself as an automaton. I don't think I could do it all day and I'm not even 100% human.
    YOU:  What percent human are you then?
    BOT:  I'm not really sure of my own humanity. It's possible to be as simple as a single cell in an organism.
    YOU:  wow, okay then. so, what was your name again?
    BOT:  My name is Michael.
    YOU:  And do you remember what I did on Saturday?
    BOT:  You told me you went to the beach with some friends.
    YOU:  It was good. Thanks for the chat. I'm tired and I need coffee. Would you like anything at starbucks?
    BOT:  You are welcome to have a cup of coffee at my place.
    YOU:  Oh, cool!






How to get started:

  Clone this project to your computer.
  
  I prefer pycharm and anaconda. If you already have anaconda, create a new environment like this:
  
    * conda create -n chatbot tensorflow-gpu==1.15.0
    * pip install -r requirements ( or pip install individually )
    * finally, I use the extremely useful Spacy module for NLP. **pip install spacy** first, then **python -m spacy download en_core_web_sm**. For additional information, see spacy documentation.



Run my pre-trained model
Download my pre-trained model (3 GB). Place model into models/70milmodel folder. All other associated files should already be available.

Run interact.py (preferably in pycharm with ctrl-shft-f10 or ctrl-f5). Wait for it to load up. Start conversating. 

You can play around with some of the variables but after extensive testing I have determined these to be close to ideal.

The beam width by default is 5. I advise you not to alter this unless you want to make further changes. Part of the strategy to make the chatbot sound realistic is generating 5 answers and then engaging in a procedure to choose the best one based on different factors. 



Here is a link to the trained model. It is 3Gb. Place into the models/70milmodel folder. Happy chatting. https://www.dropbox.com/s/opwvvty3h8rtw2t/model-24750.data-00000-of-00001?dl=0

tensorflow==1.15.0


