'''
Name:           Hunter Manhart
Email:          Hunter.M.Manhart@vanderbilt.edu
VUnet:          manharhm
Course:         CS 3891
'''

from cltk.tokenize.word import WordTokenizer
from collections import Counter
import os
import sys
import argparse
import math
import numpy as np
import tensorflow as tf


#   Folder for saving writings
current_path = os.path.dirname(os.path.realpath(sys.argv[0]))

parser = argparse.ArgumentParser()
parser.add_argument(
    '--log_dir',
    type=str,
    default=os.path.join(current_path, 'log'),
    help='The log directory for TensorBoard summaries.')
FLAGS, unparsed = parser.parse_known_args()

# Create the directory for TensorBoard variables if there is not.
if not os.path.exists(FLAGS.log_dir):
  os.makedirs(FLAGS.log_dir)



train_text_dir = '~/cltk_data/greek/text/greek_text_first1kgreek_plaintext/'
train_text_file = 'tlg0018.tlg001.opp-grc1.txt'


def download(): 
    """Download text files if not already"""
    dir_in_usr = os.path.expanduser(train_text_dir)

    if not os.path.isdir(dir_in_usr):
        print("not a directory")
        import import_greek


def read_text():
    """Read in a file from the greek texts directory"""
    word_tokenizer = WordTokenizer('greek')

    filename = train_text_dir + train_text_file
    text = os.path.expanduser(filename)

    with open(text) as f:
        r = f.read()
        return word_tokenizer.tokenize(r) # Need to remove non-greek characters


def make_vocab(words, size):
    """
    Make a vocab of the top *size* most frequent words
    word_id: map(word->id)
    id_word: map(id->word)
    count: count of most common words
    id_text: text with ids in place of words
    """
    #   Uncommon words and then common in vocab
    count = [['UNK', 0]]
    count.extend(Counter(words).most_common(size - 1))

    #   Making mapping from word to arbitrary id (here by increment)
    word_id = dict()
    for word, _ in count:
        word_id[word] = len(word_id)  # words are unique in this count

    #   Build text as word ids and add uncommon count to 
    id_text = list()
    uncommon = 0
    for word in words:
        #   0 alligns with unk in dictionary, bc first in count
        index = word_id.get(word, 0)
        if index == 0:
            uncommon += 1
        
        #   Add word's id in sequence to text with ids for words
        id_text.append(index)

    #   Set number of uncommons
    count[0][1] = uncommon
    
    #   Mapping from id to word
    id_word = dict(zip(word_id.values(), word_id.keys()))

    #   Return all
    return id_text, count, word_id, id_word


def get_sample(id_text, window):
    """Yield from the text a word and context words around it within a window"""
    for i, word in enumerate(id_text):
        for context in id_text[max(0, i - window) : i]:
            yield word, context
        for context in id_text[i+1 : i + window + 1]:
            yield word, context
    
    #   To keep yields coming forever (tail-recursive, but should randomize)
    get_sample(id_text, window)


def get_batch(id_text, batch_size, window):
    """Yields a batch of (words, contexts)"""
    samples = get_sample(id_text, window)
    while True:
        words = np.zeros(batch_size, dtype=np.int32)
        contexts = np.zeros((batch_size, 1), dtype=np.int32)

        for index in range(batch_size):
            words[index], contexts[index, 0] = next(samples)
        yield words, contexts


#   Get words from text
download()
words = read_text()

#   Define vocab size
vocab_size = 1000

#   Get mappings, count, and id text
id_text, count, word_id, id_word = make_vocab(words, vocab_size)
del words       #   Frees up memory

#   
print('Most common: ', count[:5])
print('Random words', id_text[:10], "\n", [id_word[i] for i in id_text[:10]])

words, contexts = next(get_batch(id_text, 16, 2))
for i in range(8):
    print(words[i], id_word[words[i]], '->', contexts[i, 0], id_word[contexts[i, 0]])



#   Hyperparameters
vocab_size = 50000
batch_size = 128
embed_size = 128            # dimension of the word embedding vectors
window = 1                  # the context window
num_sampled = 64            # number of negative examples to sample
rate = 1.0
steps = 100000
SKIP_STEP = 5000


def generator():
    yield from get_batch(id_text, batch_size, window)

#   Dataset from batch generator
dataset = tf.data.Dataset.from_generator(generator, 
                                (tf.int32, tf.int32), 
                                (tf.TensorShape([batch_size]), tf.TensorShape([batch_size, 1])))

#   Initialize data
with tf.name_scope('data'):
    iterator = dataset.make_initializable_iterator()
    train_words = tf.placeholder(tf.int32, shape=[batch_size])
    train_contexts = tf.placeholder(tf.int32, shape=[batch_size, 1])

#   Initialize embeded weights and lookup
with tf.name_scope('embed'):
    lookup = tf.get_variable('lookup', (vocab_size, embed_size),
                                        initializer=tf.random_uniform_initializer())
    embeded = tf.nn.embedding_lookup(lookup, train_words)

with tf.name_scope('weights'):
    # Weights for NCE loss
    nce_weights = tf.Variable(
          tf.truncated_normal(
              [vocab_size, embed_size],
              stddev=1.0 / math.sqrt(embed_size)))
    nce_biases = tf.Variable(tf.zeros([vocab_size]))

with tf.name_scope('loss'):
    # NCE loss function
    loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights, 
                                        biases=nce_biases, 
                                        labels=train_contexts, 
                                        inputs=embeded, 
                                        num_sampled=num_sampled, 
                                        num_classes=vocab_size))

with tf.name_scope('optimizer'):
    optimizer = tf.train.GradientDescentOptimizer(rate).minimize(loss)

with tf.name_scope('summaries'):
    tf.summary.scalar('loss', loss)
    summaries = tf.summary.merge_all()

# Add variable initializer.
init = tf.global_variables_initializer()

# Saves weights and embeding/lookup
saver = tf.train.Saver()


#       Training
initial_step = 0
with tf.Session() as session:
    # Open a writer to write summaries.
    writer = tf.summary.FileWriter(FLAGS.log_dir, session.graph)

    # We must initialize all variables before we use them.
    init.run()

    for step in range(0, steps):
        words, contexts = iterator.get_next()
        feed_dict = {train_words: words, train_contexts: contexts}

        # We perform one update step by evaluating the optimizer op (including it
        # in the list of returned values for session.run()
        # Also, evaluate the merged op to get all summaries from the returned "summary" variable.
        # Feed metadata variable to session for visualizing the graph in TensorBoard.
        _, summary, loss_val = session.run(
            [optimizer, summaries, loss],
            feed_dict=feed_dict)
        average_loss += loss_val

        # Add returned summaries to writer in each step.
        writer.add_summary(summary, step)
        # Add metadata to visualize the graph for the last run.
        if step == (num_steps - 1):
            writer.add_run_metadata(run_metadata, 'step%d' % step)

        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            # The average loss is an estimate of the loss over the last 2000 batches.
            print('Average loss at step ', step, ': ', average_loss)
            average_loss = 0

    writer.close()


