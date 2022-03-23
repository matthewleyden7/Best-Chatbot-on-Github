import tensorflow as tf
import numpy
from gpt_2_simple.src import model
import sys

def top_k_logits(logits, k):
    if k == 0:
        # no truncation
        return logits

    def _top_k():
        values, _ = tf.nn.top_k(logits, k=k)
        min_values = values[:, -1, tf.newaxis]
        return tf.compat.v1.where(
            logits < min_values,
            tf.ones_like(logits, dtype=logits.dtype) * -1e10,
            logits,
        )
    return tf.cond(
        pred=tf.equal(k, 0),
        true_fn=lambda: logits,
        false_fn=lambda: _top_k(),
    )



def top_p_logits(logits, p):

    with tf.compat.v1.variable_scope('top_p_logits'):
        logits_sort = tf.sort(logits, direction='DESCENDING')
        probs_sort = tf.nn.softmax(logits_sort)
        probs_sums = tf.cumsum(probs_sort, axis=1, exclusive=True)
        logits_masked = tf.compat.v1.where(probs_sums < p, logits_sort, tf.ones_like(
            logits_sort)*1000)  # [batchsize, vocab]
        min_logits = tf.reduce_min(input_tensor=logits_masked, axis=1, keepdims=True)  # [batchsize, 1]


        return tf.compat.v1.where(
            logits < min_logits,
            tf.ones_like(logits, dtype=logits.dtype) * -1e10,
            logits,
        )

def tail_free(logits, z, temperature=1.0):

    logits = logits / tf.to_float(temperature)
    sps = tf.sort(tf.nn.softmax(logits, axis=1), direction='DESCENDING', axis=1)
    grad = sps[:, 1:] - sps[:, :-1]  # first derivative

    grad = grad[:, 1:] - grad[:, :-1]  # this is the 2nd derivative

    only_pos = tf.math.abs(grad)
    sec_indices = tf.range(grad.shape[1].value)
    sec_weights = only_pos / tf.math.reduce_sum(only_pos, axis=1, keepdims=True)

    tail_ids = tf.cast(tf.argmax(tf.cast(tf.cumsum(sec_weights, axis=1) > z, tf.int8), axis=1), tf.int32) + 1
    #tail_ids = tf.divide(tf.add_n([tail_ids for i in range(250)]), 250)
    # adding one to put it in the center of the tail.

    logit_inds = tf.stack([tf.range(0, logits.shape[0].value), tail_ids], axis=1)
    tail_min_vals = tf.expand_dims(tf.gather_nd(logits, logit_inds), 1)
    tail_min_vals = tf.divide(tf.add_n([tail_min_vals for i in range(250)]), 250)



    # removes any tokens below the tail location by setting their values to be very very small.
    return tf.where(
        logits < tail_min_vals,
        tf.ones_like(logits, dtype=logits.dtype) * -1e10,
        logits,
    )

def tail_free_one(logits, z):

    sps = tf.sort(tf.nn.softmax(logits, axis=1), direction='DESCENDING', axis=1)
    grad = sps[:, 1:] - sps[:, :-1]  # first derivative

    # removed the second derivative
    #grad = grad[:, 1:] - grad[:, :-1]  # this is the 2nd derivative

    only_pos = tf.math.abs(grad)
    sec_indices = tf.range(grad.shape[1].value)
    sec_weights = only_pos / tf.math.reduce_sum(only_pos, axis=1, keepdims=True)

    tail_ids = tf.cast(tf.argmax(tf.cast(tf.cumsum(sec_weights, axis=1) > z, tf.int8), axis=1), tf.int32) + 1
    #tail_ids = tf.divide(tf.add_n([tail_ids for i in range(250)]), 250)
    # adding one to put it in the center of the tail.

    logit_inds = tf.stack([tf.range(0, logits.shape[0].value), tail_ids], axis=1)
    tail_min_vals = tf.expand_dims(tf.gather_nd(logits, logit_inds), 1)
    tail_min_vals = tf.divide(tf.add_n([tail_min_vals for i in range(250)]), 250)



    # removes any tokens below the tail location by setting their values to be very very small.
    return tf.where(
        logits < tail_min_vals,
        tf.ones_like(logits, dtype=logits.dtype) * -1e10,
        logits,
    )

def sample_sequence(*, hparams, length, start_token=None,
                    batch_size=None, context=None, temperature=1, temperaturedos=1.0,
                    top_k=40, top_p=0.0, top_l=0, top_l2=150, zed=0.9, loss=.001, loss2=.001, name='', top_p_dos=0.9, top_l3=7):
    if start_token is None:
        assert context is not None, 'Specify exactly one of start_token and context!'
    else:
        assert context is None, 'Specify exactly one of start_token and context!'
        context = tf.fill([batch_size, 1], start_token)

    def step(hparams, tokens, past=None):

        lm_output = model.model(hparams=hparams, X=tokens, past=past, reuse=tf.compat.v1.AUTO_REUSE)
        logits = lm_output['logits'][:, :, :hparams.n_vocab]

        presents = lm_output['present']

        presents.set_shape(model.past_shape(
            hparams=hparams, batch_size=batch_size))
        return {
            'logits': logits,
            'presents': presents,
        }

    with tf.compat.v1.name_scope('sample_sequence'):
        # Don't feed the last context token -- leave that to the loop below
        # TODO: Would be slightly faster if we called step on the entire context,
        # rather than leaving the last token transformer calculation to the while loop.


        context_output = step(hparams, context[:, :-1])

        def body(past, prev, output):

            next_outputs = step(hparams, prev[:, tf.newaxis], past=past)

            if name == 'question_bot':
                print('question_bot')

                logits = next_outputs['logits'][:, -1, :] / tf.cast(temperature, tf.float32)

                # pass through one top k layer
                logits = top_k_logits(logits, k=top_k)

                for i in range(top_l2):
                    logits = tf.divide(tf.add_n([top_p_logits(logits, p=top_p) for i in range(top_l)]), top_l)
                    logits = tf.keras.layers.Average()([logits, top_p_logits(logits, p=top_p_dos), logits])
                logits = tail_free_one(logits, z=zed)

            if name == 'statement_bot':
                print('statement_bot')

                logits = next_outputs['logits'][:, -1, :] / tf.cast(temperature, tf.float32)

                logits = top_k_logits(logits, k=top_k)
                for i in range(top_l2):
                    logits = tf.divide(tf.add_n([top_p_logits(logits, p=top_p) for i in range(top_l)]), top_l)
                    logits = tf.keras.layers.Average()([logits * 0.95, logits * 0.975, logits, top_p_logits(logits, p=top_p_dos), logits, logits * 0.975, logits * 0.95])
                logits = tail_free_one(logits, z=zed)


            if name == 'knowledge_bot':
                print('knowledge_bot')

                logits = next_outputs['logits'][:, -1, :] / tf.cast(temperature, tf.float32)

                logits = top_k_logits(logits, k=top_k)

                for i in range(top_l2):
                    logits = tf.divide(tf.add_n([top_p_logits(logits, p=top_p) for i in range(top_l)]), top_l)
                    logits = tf.keras.layers.Average()([logits, top_p_logits(logits, p=top_p_dos), logits])


            samples = tf.random.categorical(
                logits, num_samples=1, dtype=tf.int32)
            return [
                tf.concat([past, next_outputs['presents']], axis=-2),
                tf.squeeze(samples, axis=[1]),
                tf.concat([output, samples], axis=1)

            ]

        def cond(*args):
            return True

        _, _, tokens = tf.while_loop(
            cond=cond, body=body,
            maximum_iterations=length,
            loop_vars=[
                context_output['presents'],
                context[:, -1],
                context,

            ],
            shape_invariants=[
                tf.TensorShape(model.past_shape(
                    hparams=hparams, batch_size=batch_size)),
                tf.TensorShape([batch_size]),
                tf.TensorShape([batch_size, None]),

            ],
            back_prop=False,
        )

        return tokens