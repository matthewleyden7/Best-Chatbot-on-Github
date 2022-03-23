import time
startingup_time = time.time()
import fire
import tensorflow as tf
from brain import bot_brain
import model, mod_sample, encoder2
import sys
import json
import random
import os
import numpy as np
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import spacy
nlp = spacy.load('en_core_web_md')
import warnings
warnings.filterwarnings("ignore")


#  Best:  temperature2=1.099, temperature2dos=0.8, top_k2=17, top_p2=0.9125, top_l2=1, top_l2second=1, top_l2third=1, loss2=0, zed2=500.0, top_p_dos2=0.892,

def interact_model(model_name='70milmodel', seed=42, nsamples=5, batch_size=5, length=20,
    name='statement_bot', temperature=1.099, temperaturedos=0.8, top_k=29, top_p=0.9, top_l=1, top_lsecond=1, top_lthird=1, loss=0, zed=700.0, top_p_dos=0.896, name2='question_bot',
    temperature2=1.099, temperature2dos=0.8, top_k2=15, top_p2=0.896, top_l2=1, top_l2second=1, top_l2third=1, loss2=0, zed2=700.0, top_p_dos2=0.9, models_dir='models'):

    start = time.time()
    models_dir = os.path.expanduser(os.path.expandvars(models_dir))
    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0

    enc = encoder2.get_encoder(model_name, models_dir)
    hparams = model.default_hparams()
    with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    with tf.Session(graph=tf.Graph()) as sess:

        context = tf.placeholder(tf.int32, [batch_size, None])

        np.random.seed(seed)
        tf.set_random_seed(seed)

        output = mod_sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            name=name,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k, top_p=top_p, top_l=top_l, zed=zed, top_l2=top_lsecond, top_l3=top_lthird, loss=loss, top_p_dos=top_p_dos, temperaturedos=temperaturedos
        )

        output2 = mod_sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            name=name2,
            batch_size=batch_size,
            temperature=temperature2, top_k=top_k2, top_p=top_p2, top_l=top_l2, top_l2=top_l2second, top_l3=top_l2third,
            zed=zed2, loss=loss2, top_p_dos=top_p_dos2, temperaturedos=temperature2dos
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
        saver.restore(sess, ckpt)

        # first response is slow
        context_tokens = enc.encode('hippos are people too...')
        out = sess.run(output, feed_dict={
            context: [context_tokens for _ in range(batch_size)],
        })[:, len(context_tokens):]
        out2 = sess.run(output2, feed_dict={
            context: [context_tokens for _ in range(batch_size)],
        })[:, len(context_tokens):]


        brain = bot_brain(enc, model_name)

        print(f'Program loaded in {time.time() - startingup_time}')

        while True:

            # User enters input here
            user = input("YOU:  ")

            if not user:
                print('BOT:  Huh?')
                continue

            check = brain.add_history(user, 'you')

            if check == False:
                continue

            new_answer = 'jdjkshbdjkshebjks dhsbkdjhfjkdsjd'
            count = 0
            executions = 0
            answer = ''
            addtl_response = ''
            new_texts = {}
            spoken_sents = []
            while True:
                executions += 1


                if new_answer == '':
                    break

                if new_texts != {} and len(new_texts[new_answer]) > 1:

                    break
                if count == 5:
                    print('\nexiting for count')
                    break

                if executions == 1:
                    context_tokens, raw_text3 = brain.generate_convo()
                    context_tokens = context_tokens[-brain.max_tokens:]
                else:
                    convo_add = raw_text3 + answer
                    context_tokens = enc.encode(convo_add)
                    context_tokens = context_tokens[-brain.max_tokens:]


                new_answers = []
                new_texts = {}
                generated = 0
                for _ in range(nsamples // batch_size):
                    if '?' in user or len(user.split()) <= 2:
                        out = sess.run(output2, feed_dict={
                            context: [context_tokens for _ in range(batch_size)]
                        })[:, len(context_tokens):]
                    else:
                        out = sess.run(output, feed_dict={
                            context: [context_tokens for _ in range(batch_size)]
                        })[:, len(context_tokens):]

                    for i in range(batch_size):

                        generated += 1
                        texto = enc.decode(out[i])
                        texto_split = texto.split('\n')
                        new_answer = texto.split('\n')[0]
                        new_texts[new_answer] = texto_split
                        new_answers.append(new_answer)

                    if executions == 1:
                        new_answer = brain.find_best_answer(new_answers, user, first_response=True)
                    else:
                        new_answer = brain.find_best_answer(new_answers, user)


                answer += new_answer
                new_texts[new_answer] = [c for c in new_texts[new_answer] if c != '']

                print('\r{}'.format(answer), end='')
                sys.stdout.flush()

                count += 1


            # add bot's response to history
            brain.add_history(answer, 'bot')

            # break the flush
            print()



if __name__ == '__main__':
    fire.Fire(interact_model)
