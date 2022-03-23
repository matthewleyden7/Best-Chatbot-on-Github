import spacy
nlp = spacy.load('en_core_web_sm')
import string
from nltk import sent_tokenize
import random
import glob
import statistics
import re
import numpy as np
import json
import time
import os



class bot_brain:
    def __init__(self, enc, model):
        self.enc = enc
        self.model=model
        self.print_memory = False
        self.time_since_mem_drop = 0
        self.mem_dropper_count = 1000
        self.entities = []
        self.memory_referenced = 0
        self.memory_recall=True
        self.mem_pointer=-2
        self.knowledge_q_pointer = 0
        self.recent_thoughts = []
        self.random_thoughts = []
        self.print_parse=False
        self.print_parse2=False
        self.command_question=False
        self.print_noun = False
        self.conversation = []
        self.decoded = ''
        self.saved_user_q = []
        self.saved_bot_q = []
        self.find_answer_saved = []
        self.possible_q = ''
        self.nouns = []
        self.print_find_answer = True
        self.modified_q = ''
        self.memory_count = 0
        self.memory_drop = 50000
        self.keep_memory = True
        self.historical_similarity = True
        self.historical_index = 7
        self.random_insert = False
        self.chunks = []
        self.use_subconscious = True
        self.print_subdata = False
        self.current_subjects = {'PERSON': [], 'THING': [], 'THING2': []}
        self.current_sub_self = []
        self.current_sub_bot = []
        self.current_sub_person = []
        self.noun_recognize = []
        self.noun_association = {}
        self.find_key = []
        self.doc = ''
        self.knowledgeable_answer = None
        self.voice_check = False
        self.using_voice = False
        self.print_data = False
        self.print_thought = False
        self.print_history = False

        self.print_answers = False
        self.print_time = False
        self.print_questions = False
        self.print_most_likely = False
        self.print_toks = False
        self.history = []
        self.user_responses = []
        self.bot_responses = []
        self.full_context = []

        self.parse_count = 0
        self.max_history = 10
        self.max_history2 = 5
        self.max_history3 = 3
        self.max_tokens = 115
        self.knowledge_question = False
        self.use_knowledge = False
        self.keywords = ['what', 'where', 'when', 'why', 'who', 'how', "who's", "where's", "when's", "what's", "which", 'What', 'Where', 'When', 'Why', 'Who', 'How', "Who's", "Where's", "When's", "What's", "Which"]
        self.keywords2 = ['what', 'where', 'when', 'why', 'who', 'how', "who's", "where's", "when's", "what's", 'which', "how's", "why's"
                          'did', 'if', 'would', "wouldn't", 'is', 'do', 'can', 'does', 'so', 'are', 'have']
        self.keywords3 = ['what', 'where', 'when', 'why', 'who', 'how', "who's", "where's", "when's", "what's", 'What',
                    'Where', 'When', 'Why', 'Who', 'How', "Who's", "Where's", "When's", "What's"]
        self.keywords4 = ['did', 'if', 'would', "wouldn't", 'do', 'can', 'does', 'so', 'have', 'should', 'could',  "which", 'Did', 'If', 'Would', "Wouldn't", 'Do', 'Can', 'Does', 'So', 'Have', 'Should', "Which", 'Could']
        self.keywords5 = ['is', 'are', 'Is', 'Are']

        with open('en_verbs.json', 'r') as f:
            self.en_verbs = json.load(f)

    # starts here. user response is identified as command, knowledge question, and added to history
    def add_history(self, sent, identity):

        # raise the count to keep track of time since the last memory drop
        self.time_since_mem_drop += 1

        self.nouns = [str(tok) for tok in nlp(sent) if tok.pos_ in ['NOUN', 'PROPN'] and str(tok).lower() not in self.keywords and str(tok).lower() not in ['i', 'you', 'yours', 'your', 'his', 'her']]
        if self.print_data: print(f'brain nouns  -->  {self.nouns}')

        # here we update the memory referenced variable to reflect a recent memory popup
        if self.memory_referenced > -1:
            self.memory_referenced += 1
            if self.memory_referenced == self.max_history:
                self.memory_referenced = 0
        # reset random sent
        self.random_sent = ''
        self.random_insert = False
        self.using_random_sent = False
        self.knowledge_question = False
        self.knowledgeable_answer = ''
        self.command_question = False
        self.modified_q = ''


        sent = sent

        #print(f'identity:  {identity}')
        if identity == 'you':

            if self.knowledge_q_pointer != 0 and '?' not in sent and len(sent.split()) > 2:

                try:
                    self.history[self.knowledge_q_pointer] = self.history[self.knowledge_q_pointer][:self.history[self.knowledge_q_pointer].index('[')] + self.history[self.knowledge_q_pointer][self.history[self.knowledge_q_pointer].index('---') + 4:]
                    print(f'this is new self.history-2 -->  {self.history[self.knowledge_q_pointer]}')
                except:
                    pass
                self.knowledge_q_pointer = 0

            # requirements to be checked for knowledge question
            checker = self.check(sent)

            if checker == True:
                return False

            if sent.split()[0] == "SUB":

                new_sent = 'YOU:  ' + ' '.join(sent.split()[1:]) + '\n'
                #print(new_sent)
                self.previous_sent = 'YOU:  ' + sent[sent.index('---') + 4:] + '\n'
                self.history.append(new_sent)

                self.time_since_mem_drop = 0

                return 'memory'


            elif sent.split()[0].lower() == "random":


                if self.full_context == []:
                    print('no memories yet')
                    return True

                user_sent = ' '.join(sent.split()[1:])
                if self.print_parse: print(f'this is user sent  -->  {user_sent}')


                # to fix
                modified_context = [c for c in self.full_context if c[2][0] != [] and c[2][0][0] in ['i', 'my'] and c[2][1] != []]
                if self.print_parse:  print(f'this is modified context  -->  {modified_context}')
                rando = random.choice(modified_context)
                if self.print_parse:  print(f'this is rando  -->  {rando}')


                memory = (f'{rando[3].upper()} MEMORY:  ' + rando[1] if '?' not in sent else f'{rando[3].upper()}:  ' + rando[1])
                if self.print_parse:  print(f'this is memory  -->  {memory}')

                new_sent = 'YOU:  [' + memory + '] --- ' + user_sent + '\n'
                if self.print_parse:  print(f'this is new sent  -->  {new_sent}')

                previous_sent = 'YOU:  ' + user_sent + '\n'

                self.history.append(new_sent)

                self.time_since_mem_drop = 0

                return 'memory'




            elif sent == 're do':
                print('redoing')
                if self.history != []:

                    self.history.pop(-1)
                    self.history.pop(-1)
                    self.history.append(self.previous_sent)
                else:
                    print('history is empty')
                return True

            elif sent in ['set brakes', 'set breaks', 'Set breaks.', 'Set brakes.']:
                print('setting breaks')
                print(f'current breaks is {self.breaks}')
                try:
                    self.breaks = int(input('What would you like to set breaks to?  -->  '))
                    print(f'breaks set to {self.breaks}')
                except:
                    print('sorry, there was a problem dude')
                return True


            elif sent in ['stats', 'Stats.']:
                return 'stats'
            elif sent == 'Fix.':
                return 'output'
            elif checker == 'voice_enabled':
                return 'voice_enabled'


            # Here we determine whether or not to drop the memory from history
            if identity == 'you' and self.memory_recall == True and '?' not in sent:

                if self.time_since_mem_drop > self.mem_dropper_count:
                    print('self.time_since_mem_drop > self.mem_dropper_count')
                    random_mem = random.choice(self.full_context)
                    print(f'random_mem -> {random_mem}')
                    random_memory = random_mem[1]
                    print(f'random_memory -> {random_memory}')
                    self.history.pop(-1)
                    #self.history[-1][:6] + f'[{random_mem[-2].upper()}:  ' + random_memory + f'] --- {self.history.pop(-1)[6:]}\n'
                    print(self.history[-3:])
                    return 'memory'

                if self.print_parse2:  print("user didn't ask a follow-up question. erasing memory")
                if self.print_parse2:  print(f'historical sentence -->  {self.history[self.mem_pointer]}')
                try:
                    self.history[self.mem_pointer] = self.history[self.mem_pointer][:6] + self.history[self.mem_pointer][self.history[self.mem_pointer].index('---') + 4:]
                except:
                    pass
                if self.print_parse2:  print(f'modified historical sentence -->  {self.history[self.mem_pointer]}')
                # reset mem_pointer to -2
                self.mem_pointer = -2
                self.memory_recall = False

            elif identity == 'you' and self.memory_recall == True and '?' in sent:
                self.mem_pointer += -2
                if self.print_parse2:  print(f'this is current mem_pointer  -->  {self.mem_pointer}')


            if self.using_voice:
                new_sent = self.fix_the_sent(sent)
            else:
                new_sent = sent

            if self.use_subconscious == True:

                parser = self.parse(new_sent, identity)

                self.memory_count += 1
                if parser != '' and parser != None:



                    self.memory_sent = parser + '\n'
                    self.previous_sent = 'YOU:  ' + new_sent + '\n'


                    self.history.append(self.memory_sent)
                    self.memory_referenced += 1
                    #print('increasing memory_referenced')

                    self.time_since_mem_drop = 0

                    return 'memory'

                elif self.memory_count >= self.memory_drop:
                    self.memory_count = 0
                    if self.print_parse: print('dropping a memory')

                    modified_context = [c for c in self.full_context if
                                        c[2][0] != [] and c[2][0][0] in ['i', 'my'] and c[2][1] != []]
                    if self.print_parse: print(f'this is modified context  -->  {modified_context}')

                    # dropping a memory based on random choice
                    if modified_context != []:
                        rando = random.choice(modified_context)


                        memory = (
                            f'{rando[3].upper()} MEMORY:  ' + rando[1] if '?' not in sent else f'{rando[3].upper()}:  ' +
                                                                                               rando[1])


                        new_sent = 'YOU:  [' + memory + '] --- ' + sent + '\n'


                        self.previous_sent = 'YOU:  ' + sent + '\n'

                        self.history.append(new_sent)

                        self.time_since_mem_drop = 0

                        return 'memory'

            self.possible_q = self.find_possible_q(new_sent)
            if self.print_data: print(f'possible_q  -->  {self.possible_q}')
            if self.possible_q is not None:
                #print(f'possible q  -->  {self.possible_q}')

                self.knowledge_question = True


            else:
                fixed_sent = 'YOU:  ' + new_sent + '\n'
                self.previous_sent = fixed_sent
                self.history.append(fixed_sent)

        else:
            if self.use_subconscious == True:
                parser = self.parse(sent[6:], 'bot')
            self.find_subject(sent, identity='bot')

            fixed_sent = sent + '\n'
            self.history.append(fixed_sent)


    def remove(self, num=0):
        if num == 0:
            self.history.pop(-1)
            self.history.append(self.previous_sent)




    def find_subject(self, new_sent, identity='you'):
        #print(f'finding {identity} subject')

        doc = nlp(new_sent)
        self.chunks = [str(chunk).lower() for chunk in doc.noun_chunks]
        entities = [(str(ent), str(ent.label_)) for ent in doc.ents]

        # identity the unacceptable entities (by label) that we do not want to reference to as a subject
        unacceptable_ents = ['TIME', 'ORDINAL', 'CARDINAL', 'DATE', 'NUMERAL', 'MONEY']

        # remove those entities from entities list
        entities = [c for c in entities if c[1] not in unacceptable_ents]

        # if entities are found we set the
        #print(f'entities  -->  {entities}')
        self.entities = entities

        if entities != []:
            label = ('PERSON' if entities[-1][1] == 'PERSON' else 'THING')
            # print(label)
            if label == 'PERSON':
                self.current_subjects['PERSON'] = [entities[-1][0].lower()]
            else:
                if identity == 'you':
                    self.current_subjects['THING'] = [entities[-1][0].lower()]
                else:
                    self.current_subjects['THING2'] = [entities[-1][0].lower()]

        else:
            toks = [(str(tok), str(tok.pos_), str(tok.dep_)) for tok in doc]
            #print(f'toks  -->  {toks}')

            current_objects = [c for c in toks if
                               c[2] == 'subj' and c[2] in ['NOUN', 'PROPN'] and c[0] not in ['it', 'that',
                                                                                             'kind', 'lot',
                                                                                             'thing', 'some',
                                                                                             'it', 'they',
                                                                                             'much', 'him',
                                                                                             'her', 'that',
                                                                                             'thing', 'things',
                                                                                             'bit', 'lot',
                                                                                             'kind', 'the',
                                                                                             'you', 'your',
                                                                                             'me', 'my']]

            current_objects = [c for c in toks if
                               c[2] == 'dobj' and c[1] in ['NOUN', 'PROPN'] and c[0] not in ['it', 'that', 'kind',
                                                                                             'lot', 'thing', 'some',
                                                                                             'it', 'they',
                                                                                             'much', 'him', 'her',
                                                                                             'that', 'thing', 'things',
                                                                                             'bit', 'lot',
                                                                                             'kind', 'the', 'you',
                                                                                             'your', 'me', 'my']]
            if current_objects == []:
                current_objects = [c for c in toks if
                                   c[2] == 'pobj' and c[1] in ['NOUN', 'PROPN'] and c[0] not in ['it', 'that', 'kind',
                                                                                                 'lot', 'thing', 'some',
                                                                                                 'it', 'they',
                                                                                                 'much', 'him',
                                                                                                 'her', 'that',
                                                                                                 'thing',
                                                                                                 'things',
                                                                                                 'bit', 'lot',
                                                                                                 'kind', 'the',
                                                                                                 'you', 'your',
                                                                                                 'me', 'my']]

            if current_objects != []:
                current_objects = [c for c in current_objects if c[0] not in unacceptable_ents]
                current_objects = [current_objects[-1][0]]

                for chunk in self.chunks:
                    if current_objects[0] in chunk:
                        current_objects = [chunk]

                if identity == 'you':
                    self.current_subjects['THING'] = current_objects
                else:
                    self.current_subjects['THING2'] = current_objects

                #print(f'current user objects  -->  {current_objects}')

            #print(f'current user objects  -->  {current_objects}')

            split_words = [word.strip(string.punctuation) for word in new_sent.lower().split()]
            #print(f'user split words  -->  {split_words}')
            if 'he' in split_words or 'she' in split_words or 'him' in split_words or 'her' in split_words or 'his' in split_words or "her's" in split_words or 'hers' in split_words or "he's" in split_words and self.current_subjects['PERSON'] != []:
                #print(f'THIS IS THE SUBJECT  -->  {self.current_subjects["PERSON"]}')
                if self.current_subjects['PERSON'] != []:
                    self.modified_q = ' '.join([self.current_subjects["PERSON"][0] if split_words[i] in ['he',
                                                                                                         'she',
                                                                                                         'him',
                                                                                                         'her',
                                                                                                         'his',
                                                                                                         "her's",
                                                                                                         "hers",
                                                                                                         "he's"] else c
                                                for i, c in enumerate(split_words)]) + '?'
                #print(f'modified q -->  {self.modified_q}')
                self.random_insert = True

            elif 'they' in split_words or 'them' in split_words or 'it' in split_words or 'that' in split_words or 'those' in split_words or 'this' in split_words or 'their' in split_words and self.current_subjects['THING']:


                if self.current_subjects['THING2'] != []:
                    self.modified_q = ' '.join(
                        [self.current_subjects["THING2"][0] if split_words[i] in ['they', 'them', 'it', 'that', 'those',
                                                                                  'this', 'their'] else c
                         for i, c in enumerate(split_words)]) + '?'
                    #print(f'modified q  -->  {self.modified_q}')
                    self.random_insert = True



    def fix_the_sent(self, j):

        # fix separate letters
        j = [x for x in [word.strip(string.punctuation) for word in j.lower().split()] if x not in ['', ' ', '  ']]

        # fix single letters
        new_j = []
        singles = []
        for i, item in enumerate(j):
            if len(item) == 1 and i < len(j) - 1 and len(j[i + 1]) == 1:
                singles.append(item.upper())
            else:
                if singles != []:
                    singles.append(item)
                    new_j.append(''.join(singles))
                    singles = []
                else:
                    new_j.append(item)

        # j = ' '.join([c + j.pop(i + 1) if i < len(j) - 1 and len(c) == 1 and len(j[i + 1]) == 1 else c for i, c in enumerate(j)])

        j = ' '.join(new_j)
        doc = nlp(j)
        toks = [(str(tok), str(tok.dep_), str(tok.pos_)) for tok in doc]
        toks = [(c[0] + toks.pop(i + 1)[0], c[1], c[2]) if i < len(toks) - 1 and "'" in toks[i + 1][0] else c for i, c
                in enumerate(toks)]


        # entities
        entities = [word for word in [str(ent).split() for ent in doc.ents] for word in word]


        # split_j = [(c[0] + '.', c[1], c[2]) if 0 < i < len(toks) - 1 and toks[i + 1][0] in keywords and c[1] in ['dobj', 'pobj', 'advmod', 'amod', 'acomp'] and c[2] in ['NOUN', 'PROPN', 'ADJ', 'ADV', 'PRON'] and
        # toks[i + 2][2] not in ['PRON', 'DET']  else c for i,c in enumerate(toks)]


        if toks != []:
            # when to add a period
            split_j = [(c[0] + '.', c[1], c[2]) if 0 <= i < len(toks) - 1
                                                   and ((c[2] == 'PRON' and toks[i + 1][2] == 'PRON' and c[
                0] not in self.keywords3 and toks[i + 1][0] not in ['something'])
                                                        or (c[1] in ['dobj', 'pobj'] and c[2] in ['NOUN', 'PROPN',
                                                                                                  'PRON'] and toks[i + 1][
                                                                0] in self.keywords3 and toks[i - 1][0] not in self.keywords4 and
                                                            toks[i + 2][0] not in self.keywords4 and toks[i + 2][2] not in [
                                                                'PRON'])
                                                        or (c[1] in ['dobj', 'pobj', 'npadvmod'] and c[2] in ['NOUN',
                                                                                                              'PROPN'] and
                                                            toks[i + 1][2] == 'PRON')
                                                        or (c[2] in ['NOUN', 'PROPN'] and toks[i + 1][0] in ['let', 'if'])
                                                        or (c[1] in ['dobj', 'pobj'] and c[2] in ['NOUN', 'PROPN'] and
                                                            toks[i - 1][1] in ['det'] and toks[i - 1][2] in ['DET'] and
                                                            toks[i + 1][0] in self.keywords5)
                                                        or (c[1] in ['dobj'] and c[2] in ['NOUN'] and toks[i + 1][
                        0] in self.keywords3)
                                                        or (c[1] in ['dobj'] and c[2] in ['NOUN'] and toks[i + 1][
                        1] == 'det' and toks[i + 1][2] == 'DET')
                                                        or (c[1] in ['attr'] and c[2] in ['NOUN'] and toks[i + 1][
                        2] == 'PRON')
                                                        or (c[1] in ['attr'] and c[2] in ['NOUN'] and toks[i + 1][
                        1] == 'det' and toks[i + 1][2] == 'DET')
                                                        or (c[1] in ['advmod'] and c[2] in ['ADV'] and toks[i + 1][1] in [
                        'neg'] and toks[i + 1][2] in ['ADV'])

                                                        # after certain words
                                                        or (c[0] in ['lol', 'Lol'])
                                                        )

                       else c for i, c in enumerate(toks)]

            # when to add a comma

            split_j = [
                (c[0] + ',', c[1], c[2]) if i in [0, 1, 2] and c[0] in ['indeed', 'ugh', 'basically', 'true', 'kidding',
                                                                        'btw', 'actually', 'well', 'hello', 'hey', 'hi',
                                                                        'so', 'nope', 'no', 'oh', 'yes', 'yea', 'yeah',
                                                                        'no', 'thanks', 'alright', 'really', 'right',
                                                                        'sure', 'cool', 'nice', 'yet', 'sometimes',
                                                                        'nothing', 'wow', 'ok',
                                                                        'okay', 'welcome', 'goodbye', 'man', 'um', 'wait',
                                                                        'imo', 'goodness', 'course'] and i < len(
                    split_j) - 1 and split_j[i + 1][0] not in ['matter', 'for', 'to', 'man', 'woman', 'girl', 'dude', 'no',
                                                               'one', 'much', 'so', 'goodness',
                                                               'problem', 'really', 'no', 'wow', 'ok', 'okay', 'left',
                                                               'home', 'ah']
                                            and split_j[i + 1][2] != 'ADJ' else c for i, c in
                enumerate(split_j)]

            split_j = [(c[0] + ',', c[1], c[2]) if 0 < i < len(split_j) - 1 and (
                    (split_j[i + 1][0] in ['but'] and c[1] in ['advmod'])
                    or (split_j[i + 1][0] in ['but', 'although', 'dude', 'especially', 'no'])
                    or (c[2] in ['NOUN'] and split_j[i + 1][0] in ['not'])
                    or (split_j[i + 1][0] in ['lol', 'lolz', 'lmao', 'honestly'] or 'haha' in split_j[i + 1][0])
                    or (c[1] == 'PRON' and split_j[i + 1][1] in ['dobj'] and split_j[i + 1][2] in ['NOUN'])
                    # or (c[1] in ['pobj'] and c[2] in ['NOUN'] and split_j[i + 1][1] in ['ccomp', 'advmod'] and split_j[i + 1][2] in ['AUX', 'ADV'])
                    or (c[1] in 'attr' and c[2] in ['NOUN'] and split_j[i + 1][1] in ['advmod'] and split_j[i + 1][
                    2] in ['ADV'])
                    or (c[1] in ['dobj'] and c[2] in ['NOUN'] and split_j[i + 1][1] in ['poss'] and split_j[i + 1][
                    2] in ['DET'])
                    or (split_j[i + 1][0] in ['lol', 'lmao', 'lolz'])
                    or (c[1] in ['pobj'] and c[2] in ['NOUN', 'DET'] and split_j[i + 1][1] in ['nsubj', 'poss'] and
                            split_j[i + 1][2] in ['PRON', 'DET'] and '.' not in c[0])
                    or (c[1] in ['dobj'] and c[2] in ['NOUN'] and split_j[i + 1][1] in ['acl', 'auxpass', 'aux'] and
                            split_j[i + 1][2] in ['VERB', 'AUX'] and split_j[i + 1][0] not in self.keywords4 and split_j[i + 1][
                                0] not in self.keywords5)
                    or (c[1] in ['acomp'] and c[2] in ['SCONJ'] and split_j[i + 1][1] in ['advmod'] and split_j[i + 1][
                    2] in ['ADV'])
                    or (c[1] in ['punct'] and c[2] in ['ADJ'] and split_j[i + 1][1] in ['poss'] and split_j[i + 1][
                    2] in ['DET'])
                    or (c[1] in ['ROOT'] and c[2] in ['VERB'] and split_j[i + 1][1] in ['aux'] and split_j[i + 1][
                    2] in ['AUX'])

                    # after certain words
                    or (c[0] in ['nah'])) else c for i, c in enumerate(split_j)]

            # when to add 2 dots
            split_j = [(c[0] + '..', c[1], c[2]) if 0 < i < len(split_j) - 1 and
                                                    ((c[0] in ['so'] and split_j[i + 1][1] not in ['acomp', 'ROOT',
                                                                                                   'advmod'] and
                                                      split_j[i + 1][2] not in ['ADJ', 'DET', 'ADV'])
                                                     or (c[1] in ['attr'] and c[2] in ['NOUN'] and split_j[i + 1][1] in [
                                                                'ccomp'] and split_j[i + 1][2] in ['VERB'])
                                                     or (c[1] in ['prep'] and c[2] in ['SCONJ'] and split_j[i + 1][1] in [
                                                                'pobj'] and split_j[i + 1][2] in ['ADJ'])) else c for i, c
                       in enumerate(split_j)]

            # capitalize start of new sentence
            split_j = [(c[0][0].upper() + c[0][1:], c[1], c[2]) if i > 0 and split_j[i - 1][0][-1] == '.' else c for i, c in
                       enumerate(split_j)]

            # capitalize the i's
            split_j = [(c[0].upper(), c[1], c[2]) if c[0] == 'i' else c for i, c in enumerate(split_j)]
            final = ' '.join([c[0] for c in split_j])

            if split_j != []:


                final_split = sent_tokenize(final)


                final_split = [c for c in final_split if c != '']
                final_split = [c.split(',') if ',' in c else c for c in final_split]
                try:
                    final_split = [[c + ',' if c[-1] not in ',.?!' else c for c in item] if type(item) == list else item for
                               item in final_split]


                    final_split = [[c[1:] if c[0] == ' ' else c for c in item] if type(item) == list else item for item in
                               final_split]
                except:
                    pass
                try:
                    final_split = [[x + '?' if x.split()[0].lower() in self.keywords3 else x for x in c] if type(c) == list else c for
                                   c in final_split]
                except:
                    pass
                final_split = [x[:-1] + '?' if type(x) == str and x[-1] not in ',?!abcdefghijklmnopqrstuvwxyz' and (
                            x.split()[0] in self.keywords3 or x.split()[0] in self.keywords4 or x.split()[0] in self.keywords5) else x for x
                               in final_split]

                final_split = [
                    x + '?' if type(x) == str and x[-1] in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ' and (
                                x.split()[0] in self.keywords3 or x.split()[0] in self.keywords4 or x.split()[0] in self.keywords5) else x
                    for x in final_split]

                final_split = [' '.join(word) if type(word) == list else word for word in final_split]
                final = ' '.join(final_split)
                final = (final[:-1] + '.' if final[-1] in [','] else final)
                final = (final + '.' if final[-1] not in '?!' else final)
                #print(final)
                final = final[0].upper() + final[1:]

                #final = (final[:final.index('.')] + '.' + final[final.index('.') + 1:] if final[final.index('.') - 1] in '1234567890' and final[final.index('.') + 2] in '1234567890' else final)
                return final

            else:
                return j



    def find_possible_q(self, sent, identity='self'):
        self.possible_q = ''
        self.doc = nlp(sent)
        self.toks = [(str(tok), str(tok.pos_), str(tok.dep_)) for tok in self.doc]
        if self.print_toks: print(self.toks)

        all_nouns = [str(tok) for tok in self.doc if tok.pos_ in ["NOUN", 'PROPN', 'NUM']]
        split_raw = [word.strip(string.punctuation) for word in sent.split()]

        # check whether first word of user response in keywords2, otherwise keywords1
        self.find_key = [c for c in self.keywords2 if c == split_raw[0]]
        if self.find_key == []:
            self.find_key = list(set(self.keywords) & set(split_raw))

        if self.find_key != [] and self.possible_q == '':
            self.find_key = [self.find_key[0]]
            split2 = sent.split()

            if self.find_key[0] in split2:
                self.possible_q = ' '.join(split2[split2.index(self.find_key[0]):])
                if self.print_data: print(self.possible_q)

            elif self.find_key[0][0].upper() + self.find_key[0][1:] in split2:
                self.possible_q = ' '.join(split2[split2.index(self.find_key[0][0].upper() + self.find_key[0][1:]):])
                if self.print_data: print(self.possible_q)

            # set find_key to be lower for use later
            self.find_key = [self.find_key[0].lower()]

        if self.possible_q != '':
            self.knowledge_doc = nlp(self.possible_q)
            self.knowledge_toks = [(str(tok), str(tok.pos_), str(tok.dep_)) for tok in self.knowledge_doc]



        if self.possible_q != '' and '?' in self.possible_q and 'you' not in self.possible_q.lower() and 'your' not in self.possible_q.lower() and 'that' not in self.possible_q.lower() and 'they' not in self.possible_q.lower() \
                and ' i ' not in self.possible_q.lower() and ' my ' not in self.possible_q.lower() and len(
                self.possible_q.split()) >= 3 and all_nouns != [] and [c for c in self.knowledge_toks if c[1] == 'PRON' and c[0].lower() not in self.keywords and c[0].lower() not in self.keywords2] == []:
            if self.entities != []:

                self.knowledge_question = True

                return self.possible_q[0].upper() + self.possible_q[1:]




        if identity == 'other' and self.possible_q != '':
            if self.possible_q.lower() != sent.lower():

                return self.possible_q + '?'

        if self.possible_q != '' and '?' in sent:
            find_ents = [str(ent) for ent in nlp(self.possible_q).ents if ent.label_ not in ['NUMERAL', 'CARDINAL', 'DATE', 'TIME', 'ORDINAL']]
            #print(f'find ents  -->  {find_ents}')
            if find_ents != []:
                return self.possible_q[0].upper() + self.possible_q[1:]

    def find_best_answer(self, answers, user, first_response=False):

        if self.print_answers and first_response == True:
            print('answers')
            for c in answers:
                print(f'{c}  -->  {nlp(user).similarity(nlp(c))}')
            print()

        ################################################################################################################
        # Here we begin the process of choosing the best answer out of the 5 generated.

        # find exact duplicates. This indicates what the model was most likely trying to say and we choose this answer to display
        answer_duplicates = [c for c in answers if answers.count(c) > 1]
        if  answer_duplicates != [] and answer_duplicates[0] != '' and self.knowledge_question == True:
            answer = answer_duplicates[0]
            return answer
        else:
            # search for questions
            find_q = [c for c in answers if '?' in c and c[6:] not in [d[6:] for d in self.history[-self.max_history:]]]

            # if a question is found 1. make sure user not already asking a question to bot 2. make sure bot is not asking
            # a question 2 responses in a row
            if len(find_q) > 1 and '?' not in user and len(self.history) > 2 and '?' not in self.history[-2]:
                if len(find_q) > 1:
                    find_q = [(q, max([nlp(q).similarity(nlp(sent)) for sent in
                                       [c[6:-1] for c in self.history[-self.max_history:]]])) for q in
                              find_q]
                    find_q2 = [c[0] for c in find_q if c[1] < 0.95]

                    if find_q2 != []:
                        similarities = [nlp(q[6:]).similarity(nlp(self.history[-1][6:])) for q in
                                        find_q2]
                        answer = find_q2[similarities.index(max(similarities))]


                    else:
                        answer = random.choice(find_q)[0]

                else:
                    if self.print_data:  print(f'the len of find_q is less than 2. choose question.')
                    answer = random.choice(find_q)

            else:

                # keeping the questions to be filtered.
                answers2 = answers

                # remove hesitant answers
                answers2 = [c for c in answers2 if
                            "I'm not sure" not in c and "I don't know" not in c and "I’m just not sure" not in c and "I’m not sure" not in c and "I don’t know" not in c and "I am not really sure" not in c]

                # if user is asking a question that is not a knowledge question, grab the keyword (if exists) and search
                # for entities in the answers that match key characteristics

                if '?' in user:
                    find_ents = [[]]
                    if self.find_key != [] and self.find_key[0] in ['when', "when's"]:
                        if 'born' in user:
                            find_ents = [[(str(ent), str(ent.label_)) for ent in nlp(c[6:]).ents if ent.label_ in ['DATE']] for c in answers2]
                        elif 'time' in user:
                            find_ents = [[(str(ent), str(ent.label_)) for ent in nlp(c[6:]).ents if ent.label_ in ['TIME', 'NUMERAL', 'CARDINAL']] for c in answers2]
                        else:
                            find_ents = [[(str(ent), str(ent.label_)) for ent in nlp(c[6:]).ents if ent.label_ in ['TIME', 'DATE']] for c in answers2]

                    if self.find_key != [] and self.find_key[0] in ['where', "where's"]:
                        find_ents = [[(str(ent), str(ent.label_)) for ent in nlp(c[6:]).ents if ent.label_ in ['GPE', 'LOC', 'FAC']] for c in answers2]

                    if self.find_key != [] and self.find_key[0] in ['what', "what's"]:
                        if 'name' in user:
                            find_ents = [[(str(ent), str(ent.label_)) for ent in nlp(c[6:]).ents if ent.label_ in ['PERSON']] for c in answers2]
                        if 'age' in user:
                            find_ents = [[(str(ent), str(ent.label_)) for ent in nlp(c[6:]).ents if ent.label_ in ['NUMERAL', 'CARDINAL', 'NOMINAL']] for c in answers2]
                            #print(f'find_ents  -->  {find_ents}')
                        if 'time'in user:
                            #print('using time')
                            find_ents = [[(str(ent), str(ent.label_)) for ent in nlp(c[6:]).ents if ent.label_ in ['TIME', 'NUMERAL', 'ORDINAL']] for c in answers2]





                    if self.find_key != [] and self.find_key[0] in ['how']:

                        if 'how much' in user.lower():
                            #print(f'filtering based on key how much')
                            find_ents = [[(str(ent), str(ent.label_)) for ent in nlp(c[6:]).ents if ent.label_ in ['MONEY', 'QUANTITY']] for c in
                                         answers2]
                            if self.print_data: print(f'find_ents  -->  {find_ents}')
                        if 'how many' in user.lower():
                            #print(f'filtering based on key how many')
                            find_ents = [[(str(ent), str(ent.label_)) for ent in nlp(c[6:]).ents if ent.label_ in ['QUANTITY', 'NUMERAL', 'CARDINAL']] for c in
                                         answers2]

                    if answers2 == []:
                        answers2 = answers
                    if find_ents != [] and  [c for c in find_ents if len(c) > 0] != []:
                        if self.print_data:  print('filtering answers2 as only answers with ents.')
                        answers2 = [c for i,c in enumerate(answers2) if len(find_ents[i]) > 0 ]


                # remove answers that have memories in them
                answers2 = [c for c in answers2 if '] --' not in c]

                # remove answers that are too similar
                similarities = [max([nlp(c).similarity(nlp(x)) for c in [r[6:] for r in self.history[-self.max_history:]]]) for x in
                    answers2]
                if self.print_data: print(f'this is answers2  -->  {answers2}')
                if self.print_data: print(f'this is similarities  -->  {similarities}')

                #print(f'similarities  -->  {similarities}')
                if self.print_data: print(answers2)
                if self.print_data: print(similarities)
                answers3 = [c for c in answers2 if similarities[answers2.index(c)] < 0.75]
                if self.print_data: print(f'answers3 aka similarity of answers after removal  -->  {answers3}')

                if answers3 != []:
                    answers2 = answers3

                words = [[str(tok) for tok in nlp(sentence[6:].lower()) if
                          tok.pos_ in ['ADJ', 'NOUN', 'PROPN'] or tok.dep_ in ['dobj', 'pobj'] and str(
                              tok) not in
                          ['it', 'they', 'much', 'him', 'her', 'that', 'thing', 'things', 'bit', 'lot',
                           'kind', 'the'] and str(tok) not in self.keywords] for sentence in answers2]



                words = [[] if [x for x in ans if ans.count(x) > 1] != [] else ans for ans in words]
                if self.print_data: print(words)
                if words == []:
                    try:
                        answer = random.choice(answers2)

                    except:
                        answer = random.choice(answers)

                else:
                    if self.print_data: print(f"word strings: {[''.join(c) for c in words]}")
                    if self.print_data: print(f"longest word string: {max([''.join(c) for c in words], key=len)}")

                    words = [''.join(c) for c in words]
                    answer = answers2[words.index(max([''.join(c) for c in words], key=len))]



            if answer[6:] in [r[6:] for r in self.history[-self.max_history:]]:
                answer = random.choice(answers2)

            return answer

    def find_best_thought(self, answers, texts, user, first_response=False):
        print('finding best thought')

        similarities = [nlp(user).similarity(nlp(c)) for c in [c[6:] for c in answers]]
        if (self.print_data or self.print_answers or self.print_memory) and first_response == True:
            print('answers')
            for c in answers:
                print(f'{c}  -->  {similarities[answers.index(c)]}')
            print()

        sent = self.history[-1]

        if self.knowledge_question:
            thought = sent[sent.index('[') + 13:sent.index(']')]

        else:
            if '?' in sent:
                thought = sent[sent.index('[') + 6:sent.index(']')]
            else:
                thought = sent[sent.index('[') + 13:sent.index(']')]
        if self.print_thought or self.print_memory: (f'knowledge thought  -->  {thought}')


        answers = [c for i,c in enumerate(answers) if similarities[i] > self.max_sim]
        print(f'texts  -->  {texts}')
        texts = [c for i,c in enumerate(texts) if similarities[i] > self.max_sim]

        new_answer, new_text, new_original_answer = self.find_best_answer(answers, texts, user,
                                                                           first_response=False)

        return new_answer, new_text, new_original_answer


    # generates the final conversation list that is passed to the model.
    def generate_convo(self, addtl_response=False):

        if not addtl_response:
            if self.knowledge_question:
                #print("it's a knowledge question")
                self.conversation = self.history[-self.max_history3:]
            elif '?' in self.history[-1]:
                if self.print_parse: print('IDENTIFIED A QUESTION')
                self.conversation = self.history[-self.max_history2:]

            else:
                if self.print_parse: print('REGULAR')
                self.conversation = self.history[-self.max_history:]

            if self.historical_similarity:
                history = self.conversation

                similarities = [nlp(item[6:-1]).similarity(nlp(history[-1][6:-1])) for item in history[:-1]]

                if self.print_parse:
                    for item in list(zip(self.conversation, similarities)):
                        print(item)
                # for finding the best answer based on similarity to average of history[-max_history]

                idxes = []
                if len(similarities) > 4:
                    # for i, item in enumerate(history):
                    # print(item[6:-1] + f'  -->  {similarities[i]}')
                    for i, c in enumerate(similarities[:-1]):
                        if c < similarities[i + 1]:
                            pass
                        else:
                            idxes.append(i)
                    # print(f'idxes  -->  {idxes}')
                    if idxes != []:
                        self.conversation = self.conversation[idxes[0] + 1:]



            raw_text3 = ''.join(self.conversation)

            if self.print_parse: print(f'\nthis is raw_text3  -->  {raw_text3}\n')
            context_tokens = self.enc.encode(raw_text3)
            context_tokens = context_tokens[-self.max_tokens:]
            self.decoded = self.enc.decode(context_tokens)
            if self.print_parse: print(f'this is the length of context tokens  -->  {len(context_tokens)}')

            return context_tokens, self.decoded

        else:

            # if this is addtl response then we remove the bots previous answer and save for later. This way the bot
            # will continue to reference the previous user statement
            previous_response = self.history[-1]
            self.conversation = self.history[-self.max_history:-1]
            raw_text3 = ''.join(self.conversation)
            print(f'\nthis is raw_text3  -->  {raw_text3}\n')
            context_tokens = self.enc.encode(raw_text3)
            context_tokens = context_tokens[-self.max_tokens:]
            self.decoded = self.enc.decode(context_tokens)
            if self.print_parse: print(f'this is the length of context tokens  -->  {len(context_tokens)}')

            return context_tokens, self.decoded, previous_response





    def parse(self, user, identity='none'):

        try:
            if identity == 'bot' and self.saved_user_q != []:
                self.saved_user_q[1] = sent_tokenize(user)[0]
                self.bot_responses.append(list(self.saved_user_q))
                self.full_context.append(list(self.saved_user_q))
                self.saved_user_q = []
            if identity == 'you' and self.saved_bot_q != []:
                self.saved_bot_q[1] = sent_tokenize(user)[0]
                self.user_responses.append(list(self.saved_bot_q))
                self.full_context.append(list(self.saved_bot_q))
                self.saved_bot_q = []
        except:
            pass


        self.parse_count += 1
        keywords = ['what', 'where', 'when', 'why', 'who', 'how', "who's", "where's", "when's", "what's", ]
        if self.print_parse: print('\n' + user)
        if self.print_parse: print(f'this is identity  -->  {identity}')
        context_type = 'practice'
        split_words = [word.strip(string.punctuation) for word in user.lower().split()]
        rando = sent_tokenize(user)

        for s in rando:
            if self.print_parse: print(s)
            if len(s.split()) > 2:
                doc = nlp(s)
                """for tok in doc:
                    print([child for child in tok.children])"""
                toks = [(str(tok).lower(), str(tok.pos_), str(tok.dep_)) for tok in doc]

                if (toks[0][2] == 'ROOT' and toks[0][1] not in ['ADJ'] and toks[0][0].lower() not in ['is', 'are', 'nice']) or toks[0][0] in ['name']:
                    #print('identified a command')
                    self.knowledge_question = True
                    self.command_question = True

                toks = [c for c in toks if c[1] != 'PUNCT' and c[2] != 'PUNCT']
                full_toks = toks
                if self.print_parse: print(toks)
                subj = [c for c in toks if c[0] in ['you', 'your', 'i', 'my', 'she', 'he', 'they']]

                if subj == []:
                    subj = [c for c in toks if
                            c[1] == 'PRON' and c[0].lower() not in keywords and c[0].lower() not in ['it', 'that']]
                if subj == []:
                    subj = [c for c in toks if c[2] == 'nsubj' and c[1] == 'PROPN' and c[0].lower() not in keywords]
                if subj == []:
                    subj = [c for c in toks if c[1] == 'PROPN' and c[0].lower() not in keywords]
                if subj == []:
                    subj = [c for c in toks if c[2] == 'nsubj' and c[1] == 'NOUN' and c[0].lower() not in keywords]
                if subj == []:
                    subj = [c for c in toks if c[1] == 'NOUN' and c[2] == 'attr' and c[0].lower() not in keywords]
                if subj == []:
                    subj = [c for c in toks if c[2] == 'nsubj' and c[0].lower() not in keywords]

                """if subj == []:
                    subj = [c for c in toks if c[1] == 'NOUN' and c[0].lower() not in keywords]"""
                # print('subj -->  {}'.format(subj))

                if subj != []:
                    if len(subj) > 1:

                        find_prons = [c for c in subj if
                                      c[0].lower() in ['you', 'your', 'i', 'my', 'he', 'she', 'they']]
                        if find_prons == []:
                            # subj = [subj[0]]
                            # choosing the last subject of the multiple
                            #print('choosing last subject')
                            # need to fix
                            subj = subj[-1]
                        else:

                            if len(find_prons) > 1:
                                if '?' in s:
                                    subj = [find_prons[-1]]
                                else:
                                    subj = [find_prons[0]]
                    else:
                        subj = [subj[-1]]

                # print('subj -->  {}'.format(subj))
                root = [c for c in toks if c[2] == 'ROOT']
                if root != []:
                    if root[0][0] in ['is', 'are']:
                        root = [c for c in toks if c[1] == 'VERB']
                        if root != []:
                            root = [root[0]]
                        else:
                            root = [c for c in toks if c[2] == 'ROOT']
                try:
                    if root != []:

                        if root[0][0] in ["'s", "'re", "'m"] and subj != [] and subj[-1] != toks[-1] and toks[
                            toks.index(subj[-1]) + 1] == root[0]:
                            index = toks.index(root[0])
                            root = []
                            toks = toks[index + 1:]
                            root.extend([c for c in toks if c[1] == 'VERB'])
                            if len(root) == 0:
                                root.extend([c for c in toks if c[1] == 'ADJ'])
                            if len(root) > 1:
                                index = toks.index(root[-1])
                            toks = toks[index + 1:]
                        elif root[0][0] in ["'s", "'re", "'m"]:
                            index = toks.index(root[0])
                            root = []
                            toks = toks[index + 1:]
                            root.extend([c for c in toks if c[1] == 'VERB'])
                            if len(root) == 0:
                                root.extend([c for c in toks if c[1] == 'ADJ'])
                            if len(root) > 1:
                                index = toks.index(root[-1])
                            toks = toks[index + 1:]
                        else:
                            toks = toks[toks.index(root[-1]) + 1:]
                except:
                    pass
                prep = [c for c in toks if c[2] == 'prep']
                if prep == []:
                    prep = [c for c in toks if c[1] == 'ADP']
                if prep != []:
                    prep = [prep[0]]
                adj = [c for c in full_toks if c[1] == 'ADJ']
                if adj == []:
                    adj = [c for c in full_toks if c[1] == 'ADV']
                verbs = [c for c in toks if c[1] == 'VERB']
                if verbs != []:
                    root.extend(verbs)
                # print('subj -->  {}'.format(subj))
                dobj = [c for c in toks if c[2] == 'nsubj' and c[1] == 'PROPN' and c[0] not in subj]
                if dobj == []:
                    dobj = [c for c in toks if c[2] == 'nsubj' and c[1] == 'NOUN']
                if dobj == []:
                    dobj = [c for c in toks if c[2] == 'dobj' and c[1] == 'PROPN']
                if dobj == []:
                    dobj = [c for c in toks if c[2] == 'dobj' and c[1] == 'NOUN']
                if dobj == []:
                    dobj = [c for c in toks if c[2] == 'dobj' and c[1] == 'PRON']
                if dobj == []:
                    dobj = [c for c in toks if c[2] == 'pobj' and c[1] == 'PROPN']
                if dobj == []:
                    dobj = [c for c in toks if c[2] == 'pobj' and c[1] == 'NOUN']
                if dobj == []:
                    dobj = [c for c in toks if c[1] == 'PROPN' and c[0] not in subj]
                if dobj == []:
                    dobj = [c for c in toks if c[1] == 'NOUN']

                if dobj == []:
                    dobj = [c for c in toks if c[2] == 'dobj' and c[0].lower() not in keywords]
                if dobj == []:
                    dobj = [c for c in toks if c[2] == 'pobj' and c[0].lower() not in keywords]

                # print('---> {}'.format(dobj))

                # print('---> {}'.format(dobj))
                other_obj = [c for c in full_toks if c[1] in ['NOUN', 'PROPN']]
                chunks = []
                dobj.extend(other_obj)
                # print('extended dobj --> {}'.format(dobj))
                if len(dobj) == 1 and subj != [] and dobj[0] == subj[0]:
                    if subj[0][2] == 'nsubj' and subj[0][1] in ['NOUN', 'PROPN']:
                        dobj = []
                    else:
                        subj = []
                dobj = [c for c in dobj if c not in ['it', 'lot', 'again']]

                new_dobj = []
                for chunk in doc.noun_chunks:
                    new_dobj = [(str(chunk).lower(), 'fixed obj') if x[0] in str(chunk).lower() and len(
                        str(chunk).lower().split()) > 1 else x for x in dobj]
                if new_dobj != []:
                    dobj.extend(new_dobj)
                dobj = list(set(dobj))
                dobj = [c for c in dobj if c not in ['it', 'lot', 'again']]
                root = [c for c in root if c not in dobj]
                # print('new dobj --> {}'.format(dobj))






                current_sub = []
                #print(f'current subjects  -->  {self.current_subjects}')
                if 'he' in split_words or 'she' in split_words or 'him' in split_words or 'her' in split_words:
                    current_sub.extend(self.current_subjects['PERSON'])


                if 'they' in split_words or 'them' in split_words or 'it' in split_words or "it's" in split_words or 'that' in split_words:
                    current_sub.extend(self.current_subjects['THING'])
                    current_sub.extend(self.current_subjects['THING2'])

                # remove non-specific objects from dobj
                to_remove = ['them', 'they', 'you', 'things', 'kind', 'stuff', 'thing', 'lot', 'it', 'they', 'that', 'time', 'one', 'ones']

                dobj = [c[0].lower() for c in dobj if c[0].lower() not in to_remove]

                # here I modify certain phrases in the dobj list
                dobj = [c[4:] if c.startswith('the ') else c for c in dobj]
                dobj = [c[3:] if c.startswith('my ') else c for c in dobj]
                dobj = [c[2:] if c.startswith('a ') else c for c in dobj]
                #print(f'this is current sub  -->  {current_sub}')
                sov = [[c[0].lower() for c in subj], [c[0].lower() for c in root], [c[0].lower() for c in prep],
                       [c[0].lower() for c in adj if c[0].lower() not in self.keywords], [c for c in dobj], [c.lower() for c in chunks],
                       current_sub]


                """if current_sub != []:
                    print(f'this is current sub:  {current_sub}')"""

                if self.print_parse: print('----------->    Subj, root, prep, adj, dobj, chunks, current_sub')

                sov[1] = [c for c in sov[1] if c not in ['do']]
                if self.print_parse2:  print('\n\nSOV ------->   {}'.format(sov))      #######



                # begin memory search


                self.find_subject(s)


                find_answer = []

                sov[4] = [c for c in sov[4] if
                          c not in ['i', 'you', 'my', 'your', 'me', 'we', 'her', 'he', 'she', 'it', 'that', 'they',
                                    'It', 'lot', 'kind', 'kinds', 'part', 'parts', 'type', 'types', 'lot', 'sorry',
                                    'thanks', 'stuff', 'shit', 'favorite', 'favourite', 'dude', 'man', 'homie',
                                    'just', 'something', 'what', 'people', 'person', 'way', 'opinion', 'thought', 'thoughts', 'place, places', 'work', 'day']]


                # attempt to find matching object
                if sov[4] != []:

                    # this is temporary to find which words it is landing on
                    """for word in sov[4]:
                        if self.root_word_conversion(word) in self.allNOUNSYMSKEYS:
                            print(f'the word {word} was identified in self.allNOUNSYMSKEYS')
                            print(f'the value is {self.allNOUNSYMS[self.root_word_conversion(word)]}')"""

                    if self.print_parse2:  print('beginning object search')
                    # first we take the dobj list and compare to historical lists (large), then we expand it. not sure if this is the best way to do it.

                    find_answer.extend([item9 for item9 in self.full_context if any(x in item9[2][4] for x in sov[4])])

                    #print(f'this is find answer after object search -> {find_answer}')


                    # we increase the size of the objects list search by including synonyms of words
                    #sov[4].extend([word for word in [ [x] + self.allNOUNSYMS[self.root_word_conversion(x)][:5] if self.root_word_conversion(x) in self.allNOUNSYMSKEYS  else [x] for x in sov[4]] for word in word])

                    sov[4] = [c for c in sov[4] if
                              c not in ['i', 'you', 'my', 'your', 'me', 'we', 'her', 'he', 'she', 'it', 'that', 'they',
                                        'It', 'lot', 'kind', 'kinds', 'part', 'parts', 'type', 'types', 'lot', 'sorry',
                                        'thanks', 'stuff', 'shit', 'favorite', 'favourite', 'dude', 'man', 'homie',
                                        'just', 'something', 'what', 'people', 'person', 'way', 'opinion', 'thought',
                                        'thoughts', 'place', 'places', 'work']]




                if '?' not in s:

                    if sov[0] != [] and sov[0][0] not in ['it', 'that'] and sov[1] != [] and sov[4] != []:
                        # increase the possible objects
                        multiples = [c + 's' if c[-1] != 's' else c[:-1] for c in sov[4]]
                        # multiples2 = [c + "'s" for c in sov[4]]
                        sov[4].extend(multiples)
                        # sov[4].extend(multiples2)
                        self.full_context.append([user, s, sov, identity, self.parse_count])

                        if identity == 'you' and self.command_question == False:
                            self.user_responses.append([user, s, sov, identity, self.parse_count])
                        elif identity == 'bot' and self.command_question == False:
                            self.bot_responses.append([user, s, sov, identity, self.parse_count])

                # The user asks the bot a question and there is no answer, if the question is about the bot, we need to
                # collect the answer
                if '?' in user and identity == 'you' and sov[0] != [] and sov[1] != []:
                    self.saved_user_q = [s, '', sov, 'bot', self.parse_count]
                if '?' in user and identity == 'bot' and sov[0] != [] and sov[1] != []:
                    self.saved_bot_q = [s, '', sov, 'you', self.parse_count]


                if identity == 'you' and (self.memory_referenced == 0 or '?' in s):

                    # If the user asks a question, we need to scan the previous statements to see if the answer is there

                    # Here we remove any aging recent thoughts
                    if self.print_parse2:  print(f'len recent thoughts before:  {len(self.recent_thoughts)}')
                    self.recent_thoughts = [c for c in self.recent_thoughts if c[-1] < self.parse_count - self.max_history]
                    if self.print_parse2:  print(f'len recent thoughts after:  {len(self.recent_thoughts)}')



                    # retrieve verb variations
                    find_verbs = []
                    if self.print_parse2:  print(f'this is sov 4 -----------------> {sov}')
                    #if sov[1] != [] and sov[4] == []:
                    for item in [c for c in sov[1] if c not in ['is', 'are']]:
                        find_verbs.append([verblist for verblist in self.en_verbs if item in verblist])
                        find_verbs = [c for c in find_verbs if c != []]
                        if self.print_parse2:  print(f'FIND VERBS  -->  {find_verbs}')
                        if find_verbs != [] and len(find_verbs[-1]) > 0:
                            find_verbs = [c[0] if type(c[0]) == list else c for c in find_verbs]

                        find_verbs = [word for word in find_verbs for word in word]

                        # remove any words that are only one letter (fix)
                        find_verbs = [c for c in find_verbs if len(c) > 1]
                        if self.print_parse2:  print(f'this is find_verbs  -->  {find_verbs}')


                    if self.print_parse2:  print(f'sov ->   {sov}')

                    # We have not found an answer with object search and now we proceed to search by verb.
                    # We verify that user asked a question before proceeding to determine best memory



                    # remove any recent thoughts to avoid repetition unless a direct question
                    if '?' not in s:
                        find_answer = [c for c in find_answer if c not in self.random_thoughts and c not in self.recent_thoughts]

                    if find_verbs != [] and sov[4] == [] and sov[3] == [] and '?' in s:
                        if self.print_parse2:  print('searching by verb')
                        find_answer.extend([c for c in self.full_context if any(item in c[2][1] for item in find_verbs) and any(item in c[2][0] for item in sov[0])])

                        if self.print_parse2:  print(f'find answer  -->  {find_answer}')


#########################################################
                        if find_answer != [] and sov[0] != [] and sov[0][0] in ['you', 'your', 'my', 'i']:
                            if self.print_parse2:  print('filtering by pronoun')
                            find_answer = [c for c in find_answer if c[2][0] != [] and c[2][0][0] in ['you', 'your', 'my', 'i']]
                            if self.print_parse2:  print(f'find_answer after filter:  {find_answer}')

                        else:

                            if sov[0] != []:
                                if self.print_parse2:  print('filtering by subject')
                                find_answer = [c for c in find_answer if c[0] != [] and c[0][0] == sov[0][0]]
                                if self.print_parse2:  print(f'find_answer after filter:  {find_answer}')


                    if len(find_answer) > 1:

                        find_answer2 = [c for c in find_answer if c[2][6] != [] and c[2][6][0] in sov[4]]
                        if find_answer2 != []:
                            print(f'there was a match for entity')
                            find_answer = find_answer2
                        else:
                            refs = [len([word for word in sov[4] if word in c[2][4]]) for c in find_answer]
                            print(f'this is refs -> {refs}')
                            if [num for num in refs if num > 1] != []:
                                print('assigning the response with most nouns')
                                find_answer = [find_answer[refs.index(max(refs))]]
                            print(f'this is new find answer -> {find_answer}')




                    if len(find_answer) > 1 and sov[1] != []:
                        if self.print_parse2:  print('\nfiltering by verb')
                        if find_verbs != []:
                            find_answer2 = [c for c in find_answer if any(item in c[2][1] for item in find_verbs) ]
                        else:
                            find_answer2 = [c for c in find_answer if any(item in c[2][1] for item in sov[1])]
                        if self.print_parse2:  print(f'find_answer2: {find_answer2}')
                        if find_answer2 != []:
                            find_answer = find_answer2


                    if find_verbs == [] and len(find_answer) > 1 and sov[3] != []:
                        if self.print_parse2:  print('trying to find interest')
                        find_verbs = [word for word in [wordlist for wordlist in self.en_verbs if any(item in sov[3] for item in wordlist)] for word in word]
                        if self.print_parse2:  print(f'new find_verbs:  {find_verbs}')
                        if find_verbs != []:
                            find_answer2 = [c for c in find_answer if any(item in c[2][1] for item in find_verbs) and any(item in c[2][3] for item in find_verbs)]
                            if find_answer2 != []:
                                find_answer = find_answer2
                            if self.print_parse2:  print(f'new find_answer:  {find_answer}')



                    # remove responses that are too recent
                    find_answer = [c for c in find_answer if c[-1] < self.parse_count - self.max_history]

                    if find_answer == [] and sov[0] != [] and sov[1] != [] and sov[4] ==[] and sov[3] != []:
                        if self.print_parse2:  print('searching by adjective')
                        find_answer = [c for c in self.full_context if c[2][0] == sov[0] and any(item in c[2][3] for item in sov[3])]
                    if find_answer != []:

                        if self.print_noun:  print(f'matching noun:  {[[item8 for item8 in sov[4] if any(x in sov[4] for x in item10[2][4])] for item10 in self.full_context]}')



                        if '?' not in s:
                            find_answer = [random.choice(find_answer)]

                        # reset the parse counts for each item that will be returned from memory
                        if find_answer != []:
                            thoughts = [c[1] for c in find_answer]

                            #self.full_context = [c[:-1] + [self.parse_count] if c[1] in thoughts else c for c in self.full_context]

                            # if multiple responses returned, filter based on verb
                            if len(find_answer) > 1 and find_verbs != []:
                                if self.print_parse2:  print('the length of find answer is greater than 1')
                                if self.print_parse2:  print(f'find answer  -->  {find_answer}')
                                find_answer2 = [c for c in find_answer if any(item in c[2][1] for item in find_verbs)]
                                if self.print_parse2:  print(f'find answer 2  -->  {find_answer2}')
                                if find_answer2 != []:
                                    find_answer = find_answer2

                            if '?' in s:
                                find_prons = [c.lower() for c in split_words if
                                              c.lower() in ['you', 'your', 'my', 'me', 'i', "i'm", "i'll"]]
                                if self.print_parse2:  print(f'find_prons:  {find_prons}')

                                if find_prons != [] and find_prons[-1] in ['i', "i'm", "i'll", 'my', 'me']:

                                    """if ('you' in find_prons or 'your' in find_prons) and not any(
                                            item in find_prons for item in ['my', 'me', 'i', "i'm", "i'll"]):"""
                                    if self.print_parse2:  print('WWWWWWWWWWWWWWWWWWWWK')


                                    """find_answer = ([c for c in find_answer if c[-2] == 'you'] if len(
                                        [c for c in find_answer if c[-2] == 'you']) > 0 else [c for c in find_answer])"""

                                    find_answer = [c for c in find_answer if c[-2] == 'you']

                                elif find_prons != [] and find_prons[-1] in ['you', 'your']:

                                    if self.print_parse2:  print('PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP')

                                    find_answer = [c for c in find_answer if c[-2] == 'bot']

                                    if len(find_answer) > 1:
                                        if self.print_parse2:  print('the len find answer is still greater than 1')
                                        find_answer = ([c for c in find_answer if c[2][0] != [] and c[2][0][0] in ['my', 'i', "i'm", "i'll"]] if len([c for c in find_answer if c[2][0] != [] and c[2][0][0] in ['my', 'i', "i'm", "i'll"]]) > 0 else [c for c in find_answer])


                        # shortened retrieved responses if too lengthy
                        find_answer = [noice[1:] if len(noice[0]) > 150 or '?' in noice[0] else noice for noice in find_answer]

                        # remove answers that are duplicates
                        find_answer2 = []
                        scripts = []

                        for answer in find_answer:
                            if answer[0] not in scripts:
                                find_answer2.append(answer)
                            scripts.append(answer[0])

                        self.find_answer_saved = find_answer2


                        #if self.print_find_answer:  print(find_answer2)
                        find_answer2 = [c for c in find_answer2 if c != []]

                        # This will display the internal thought of the bot
                        if find_answer2 != []:  print(f'This is thoughts:  {[c[1] for c in find_answer]}')

                        if '?' in user and find_answer2 != []:

                            answer_string = ' '.join([memory[-2].upper() + ':  ' + (memory[0][:-1] if memory[0][-1] == '\n' else memory[0]) + '.' for memory in find_answer2])

                            self.recent_thoughts.extend([memory for memory in find_answer2])
                            if self.print_parse2:  print()
                            if self.print_parse2:
                                for item in find_answer2:
                                    print(item)
                            if self.print_parse2:  print()


                            addtl_subjects = [memory[2] if type(memory[2]) == list else memory[1] for memory in find_answer2]
                            if self.print_parse2:  print(f'\naddtl_subjects  -->  {addtl_subjects}')
                            if addtl_subjects != []:
                                addtl_subjects = [item for item in [c[6] if len(c[6]) > 0 else '' for c in addtl_subjects] if item != '']
                            if addtl_subjects != []:
                                if self.print_parse2: print(f'\naddtl_subjects  -->  {addtl_subjects}')

                                answer_string = answer_string[:6] + ', '.join([addtl[0] for addtl in addtl_subjects]) + ', ' + answer_string[6:]
                                if self.print_parse2:  print(f'revised answer string:  {answer_string}\n')
                            self.memory_count = 0
                            self.memory_recall = True

                            # user can set print_memory to true to see bot's thoughts
                            if self.print_memory:  print(f'\n\nTHIS IS A MEMORY -> {answer_string[6:]}\n\n')

                            return f'YOU:  [{answer_string}] --- {user}'

                        # We want to verify that
                        elif '?' not in user and find_answer2 != []:
                            if self.print_parse2:  print('\nchoosing a random memory\n')
                            if len(find_answer2) > 1:
                                find_answer2 = [random.choice(find_answer2)]

                            answer_string = ' '.join([memory[-2].upper() + ' MEMORY:  ' + (memory[0][:-1] if memory[0][-1] == '\n' else memory[0]) + '.' for memory in find_answer2])
                            self.memory_count = 0
                            self.memory_recall = True
                            self.random_thoughts.extend([memory for memory in find_answer2])
                            self.recent_thoughts.extend([memory for memory in find_answer2])
                            if self.print_parse2:  print(f'this is self.recent_thoughts:  {self.recent_thoughts}')
                            # print(f'answer_string  -->  {answer_string}')

                            # here we speak
                            if self.using_voice or self.voice_check:
                                if self.speak_memory:
                                    self.botspeak(answer_string[6:], v_name="Matthew")
                                elif self.print_memory:
                                    print(f'\n\nTHIS IS A MEMORY -> {answer_string[6:]}\n\n')

                            return f'YOU:  [{answer_string[6:]}] --- {user}'


    def check(self, sent):


        if sent == 'print data':
            if self.print_data == True:
                self.print_data = False
            else:
                self.print_data = True
            print('print_data is set to {}'.format(self.print_data))
            return True

        elif sent == 'clear':
            self.history = []
            self.user_responses = []
            self.bot_responses = []
            self.full_context = []
            self.noun_association = {}
            self.noun_recognize = []
            self.q_asked = 0
            self.q_correct = 0
            self.rating = 0

            print('history cleared')
            return True

        elif sent == 'print answers':
            if self.print_answers == True:
                self.print_answers = False
            else:
                self.print_answers = True

            print('print_answers is set to {}'.format(self.print_answers))
            return True

        elif sent in ['print history', 'Print history.']:
            if self.print_history == True:
                self.print_history = False
            else:
                self.print_history = True
            print('print_history is set to {}'.format(self.print_history))
            return True

        elif sent in ['history', 'History.']:
            print(self.decoded)
            return True

            return True

        elif 'full history' in sent or 'Full history' in sent:
            print(''.join(self.history))
            return True

        elif sent in ['print memory', 'Print memory.']:
            if self.print_memory == True:
                self.print_memory = False
            else:
                self.print_memory = True
            print(f'print_memory set to {self.print_memory}')
            return True

        elif sent in ['erase', 'Erase.', 'Eras.', 'Iras.', 'He eris.', 'Iris.']:
            if self.history != []:
                self.history.pop(-1)
                self.history.pop(-1)
                print('reseting the memory recall and mem_pointer')
                print(f'this is the current parsecount -> {self.parse_count}')
                self.full_context = [c for c in self.full_context if c[-1] not in [self.parse_count, self.parse_count - 1]]
                self.user_responses = [c for c in self.user_responses if c[-1] not in [self.parse_count, self.parse_count - 1]]
                self.bot_responses = [c for c in self.bot_responses if c[-1] not in [self.parse_count, self.parse_count - 1]]
                self.memory_recall=False
                self.mem_pointer=-2
                self.find_answer_saved = []
                if self.using_voice:
                    self.engine.say('popped.')
                    self.engine.runAndWait()

                return True

            else:
                print('history empty, cannot pop from an empty list')
                if self.using_voice:
                    self.engine.say('history empty.')
                    self.engine.runAndWait()

                return True

        elif sent == 'max':
            print(f'current max history  -->  {self.max_history}')
            q = input('What would you like to set max_history to? >>>  ')
            try:
                self.max_history = int(q)
                print('max_history has been set to {}'.format(q))

            except:
                print('you entered an invalid selection')
            return True

        elif sent == 'max tokens':
            print(f'current max tokens -> {self.max_tokens}')
            num_tokens = input('What would you like to set max_tokens to?  -->  ')
            try:
                self.max_tokens = int(num_tokens)
            except:
                num_tokens = input('You did not enter a number. What would you like to set max_tokens to?  -->  ')
            print(f'max tokens set to {self.max_tokens}')
            return True

        elif sent == 'print':
            with open('saved_convos.txt', encoding='utf-8') as f:
                lines = f.readlines()
            lines = [c for c in lines if 'Conversation #' in c]
            if lines != []:
                last_conversation_num = int(lines[-1].split()[-1])
                last_conversation_num += 1
            else:
                last_conversation_num = 1
            with open('saved_convos.txt', 'a', encoding='utf-8') as f:
                f.write("Conversation # {}\n".format(last_conversation_num))

                for c in self.history:
                    f.write(c)
                f.write('\n')
                f.write('\n')
            print(f'Conversation # {last_conversation_num} has been added.')
            if self.using_voice:
                self.engine.say(f'Conversation # {last_conversation_num} has been added.')
                self.engine.runAndWait()
            return True

        elif sent in ['print responses', 'Print responses.']:
            print('user responses')
            for c in self.user_responses:
                print(c)
            print(f'len user responses:  {len(self.user_responses)}')
            print()
            print('bot responses')
            for c in self.bot_responses:
                print(c)
            print(f'len bot responses:  {len(self.bot_responses)}')
            print()
            for c in self.full_context:
                print(c)
            print(f'len full context:  {len(self.full_context)}')
            print()
            return True

        elif sent in ['use sub', 'Use sub.']:
            if self.use_subconscious == True:
                self.use_subconscious = False
            else:
                self.use_subconscious = True
            print(f'use subconscious is set to {self.use_subconscious}')


        elif sent == 'print parse':
            if self.print_parse == True:
                self.print_parse = False
            else:
                self.print_parse = True
            print(f'print_parse set to {self.print_parse}')
            return True

        elif sent == 'print parse2':
            if self.print_parse2 == True:
                self.print_parse2 = False
            else:
                self.print_parse2 = True
            print(f'print_parse2 set to {self.print_parse2}')
            return True

        else:
            return False

