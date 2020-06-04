import os
import sys
import json
import numpy as np
import conlleval

from collections import deque, namedtuple
from argparse import ArgumentParser

os.environ['TF_KERAS'] = '1'

from tensorflow import keras
#import keras
from bert import tokenization
from keras_bert import load_trained_model_from_checkpoint, AdamWarmup
from keras_bert import calc_train_steps, get_custom_objects

from config import DEFAULT_SEQ_LEN, DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS
from config import DEFAULT_LR, DEFAULT_WARMUP_PROPORTION, DEFAULT_PREDICT_START, DEFAULT_NUM_GPUS

Sentences = namedtuple('Sentences', [
    'words', 'tokens', 'labels', 'lengths', 
    'combined_tokens', 'combined_labels','sentence_numbers', 'sentence_starts'
])


def argument_parser(mode='train'):
    argparser = ArgumentParser()
    if mode == 'train':
        argparser.add_argument(
            '--train_data', required=True,
            help='Training data'
        )
        argparser.add_argument(
            '--dev_data', default=None,
            help='Development data'
        )
        argparser.add_argument(
            '--vocab_file', required=True,
            help='Vocabulary file that BERT model was trained on'
        )
        argparser.add_argument(
            '--bert_config_file', required=True,
            help='Configuration for pre-trained BERT model'
        )
        argparser.add_argument(
            '--init_checkpoint', required=True,
            help='Initial checkpoint for pre-trained BERT model'
        )
        argparser.add_argument(
            '--max_seq_length', type=int, default=DEFAULT_SEQ_LEN,
            help='Maximum input sequence length in WordPieces'
        )
        argparser.add_argument(
            '--do_lower_case', default=False, action='store_true',
            help='Lower case input text (for uncased models)'
        )
        argparser.add_argument(
            '--learning_rate', type=float, default=DEFAULT_LR,
            help='Initial learning rate'
        )
        argparser.add_argument(
            '--num_train_epochs', type=int, default=DEFAULT_EPOCHS,
            help='Number of training epochs'
        )
        argparser.add_argument(
            '--warmup_proportion', type=float, default=DEFAULT_WARMUP_PROPORTION,
            help='Proportion of training to perform LR warmup for'
        )
        argparser.add_argument(
            '--use_ner_model', default=False, action='store_true',
            help = 'use pretrained NER model instead of training a new one'
        )
        argparser.add_argument(
            '--num_gpus', type=int, default=DEFAULT_NUM_GPUS,
            help = 'Number of GPUs to use for multi-gpu'
        )
    if mode != 'serve':
        argparser.add_argument(
            '--test_data', required=True,
            help='Test data'
    )
    argparser.add_argument(
        '--predict_position', type=int, default=DEFAULT_PREDICT_START,
        help='The starting position of the predicted sentence'
    )
    argparser.add_argument(
        '--batch_size', type=int, default=DEFAULT_BATCH_SIZE,
        help='Batch size for training'
    )
    argparser.add_argument(
        '--output_file', default="output",
        help='File to write predicted outputs to'
    )
    argparser.add_argument(
        '--ner_model_dir', default='ner-model',
        help='Trained NER model directory'
    )
    argparser.add_argument(
        '--no_context', default=False, action='store_true',
        help='Do not use context in training or predict'
    ) 
    argparser.add_argument(
        '--documentwise', default=False, action='store_true',
        help='Make samples documentwise'
    )


    return argparser

def load_pretrained(options):
    model = load_trained_model_from_checkpoint(
        options.bert_config_file,
        options.init_checkpoint,
        training=False,
        trainable=True,
        seq_len=options.max_seq_length,
    )
    tokenizer = tokenization.FullTokenizer(
        vocab_file=options.vocab_file,
        do_lower_case=options.do_lower_case
    )
    return model, tokenizer


def _ner_model_path(ner_model_dir):
    return os.path.join(ner_model_dir, 'model.hdf5')


def _ner_vocab_path(ner_model_dir):
    return os.path.join(ner_model_dir, 'vocab.txt')


def _ner_labels_path(ner_model_dir):
    return os.path.join(ner_model_dir, 'labels.txt')


def _ner_config_path(ner_model_dir):
    return os.path.join(ner_model_dir, 'config.json')


def save_ner_model(ner_model, tokenizer, labels, options):
    os.makedirs(options.ner_model_dir, exist_ok=True)
    config = {
        'do_lower_case': options.do_lower_case,
        'max_seq_length': options.max_seq_length,
    }
    with open(_ner_config_path(options.ner_model_dir), 'w') as out:
        json.dump(config, out, indent=4)
    ner_model.save(_ner_model_path(options.ner_model_dir))
    with open(_ner_labels_path(options.ner_model_dir), 'w') as out:
        for label in labels:
            print(label, file=out)
    with open(_ner_vocab_path(options.ner_model_dir), 'w') as out:
        for i, v in sorted(list(tokenizer.inv_vocab.items())):
            print(v, file=out)


def load_ner_model(ner_model_dir):
    with open(_ner_config_path(ner_model_dir)) as f:
        config = json.load(f)
    model = keras.models.load_model(
        _ner_model_path(ner_model_dir),
        custom_objects=get_custom_objects()
    )
    tokenizer = tokenization.FullTokenizer(
        vocab_file=_ner_vocab_path(ner_model_dir),
        do_lower_case=config['do_lower_case']
    )
    labels = read_labels(_ner_labels_path(ner_model_dir))
    return model, tokenizer, labels, config


def read_labels(path):
    labels = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line in labels:
                raise ValueError('duplicate value {} in {}'.format(line, path))
            labels.append(line)
    return labels


def create_ner_model(pretrained_model, num_labels):
    ner_inputs = pretrained_model.inputs[:2]
    ner_output = keras.layers.Dense(
        num_labels,
        activation='softmax'
    )(pretrained_model.output)
    ner_model = keras.models.Model(inputs=ner_inputs, outputs=ner_output)
    return ner_model


def create_optimizer(num_example, options):
    total_steps, warmup_steps = calc_train_steps(
        num_example=num_example,
        batch_size=options.batch_size,
        epochs=options.num_train_epochs,
        warmup_proportion=options.warmup_proportion,
    )
    optimizer = AdamWarmup(
        total_steps,
        warmup_steps,
        lr=options.learning_rate,
        epsilon=1e-6,
        weight_decay=0.01,
        weight_decay_pattern=['embeddings', 'kernel', 'W1', 'W2', 'Wk', 'Wq', 'Wv', 'Wo'],
        clipnorm=1.0
    )
    return optimizer



def encode(lines, tokenizer, max_len):
    tids = []
    sids = []
    for line in lines:
        tokens = ["[CLS]"]+line
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        segment_ids = [0] * len(token_ids)
        if len(token_ids) < max_len:
            pad_len = max_len - len(token_ids)
            token_ids += tokenizer.convert_tokens_to_ids(["[PAD]"]) * pad_len
            segment_ids += [0] * pad_len
        tids.append(token_ids)
        sids.append(segment_ids)
    return [np.array(tids), np.array(sids)]

def label_encode(labels, tag_dict, max_len):
    encoded = []
    sample_weights = []
    for sentence in labels:
        enc = [tag_dict[i] for i in sentence]
        enc.insert(0, tag_dict['O'])
        weight = [0 if i=='[SEP]' else 1 for i in sentence]
        weight.insert(0,0) # [CLS] token weight
        if len(enc) < max_len:
            weight.extend([0]*(max_len-len(enc)))
            enc.extend([tag_dict['O']]*(max_len-len(enc)))
        encoded.append(np.array(enc))
        sample_weights.append(np.array(weight))
    lab_enc = np.expand_dims(np.stack(encoded, axis=0), axis=-1)
    weights = np.stack(sample_weights, axis=0)
    return lab_enc, weights


def get_labels(label_sequences):
    unique = set([t for s in label_sequences for t in s])
    labels = sorted(list(unique), reverse=True)
    return labels


def read_tags(path):
    f = open(path, 'r')
    tags = set(l.split()[1] for l in f if l.strip() != '')
    return {tag: index for index, tag in enumerate(tags)}


def tokenize_and_split(words, word_labels, tokenizer, max_length):
    # Tokenize each word in sentence, propagate labels
    tokens, labels, lengths = [], [], []
    for word, label in zip(words, word_labels):
        tokenized = tokenizer.tokenize(word)
        tokens.extend(tokenized)
        lengths.append(len(tokenized))
        for i, token in enumerate(tokenized):
            if i == 0:
                labels.append(label)
            else:
                if label.startswith('B'):
                    labels.append('I'+label[1:])
                else:
                    labels.append(label)

    # Split into multiple sentences if too long
    split_tokens, split_labels = [], []
    start, end = 0, max_length
    while end < len(tokens):
        # Avoid splitting inside tokenized word
        while end > start and tokens[end].startswith('##'):
            end -= 1
        if end == start:
            end = start + max_length    # only continuations
        split_tokens.append(tokens[start:end])
        split_labels.append(labels[start:end])
        start = end
        end += max_length
    split_tokens.append(tokens[start:])
    split_labels.append(labels[start:])

    return split_tokens, split_labels, lengths


def tokenize_and_split_sentences(orig_words, orig_labels, tokenizer, max_length):
    words, labels, lengths = [], [], []
    for w, l in zip(orig_words, orig_labels):
        split_w, split_l, lens = tokenize_and_split(w, l, tokenizer, max_length-2)
        words.extend(split_w)
        labels.extend(split_l)
        lengths.extend(lens)
    return words, labels, lengths


def read_sentences(input_file):
    sentences, words = [], []
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                words.append(line.split('\t')[0])
            elif words:
                sentences.append(words)
                words = []
    if words:
        sentences.append(words)
    return sentences
    

def read_conll(input_file, mode='train'):
    # words and labels are lists of lists, outer for sentences and
    # inner for the words/labels of each sentence.
    words, labels = [], []
    curr_words, curr_labels = [], []
    with open(input_file) as f:
        for line in f:
            line = line.strip()
            if line:
                fields = line.split('\t')
                if len(fields) > 1:
                    curr_words.append(fields[0])
                    if mode != 'test':
                        curr_labels.append(fields[1])
                    else:
                        curr_labels.append('O')
                else:
                    print('ignoring line: {}'.format(line), file=sys.stderr)
                    pass
            elif curr_words:
                words.append(curr_words)
                labels.append(curr_labels)
                curr_words, curr_labels = [], []
    if curr_words:
        words.append(curr_words)
        labels.append(curr_labels)
    return words, labels


def process_sentences(words, orig_labels, tokenizer, max_seq_len, seq_start=0):
    # Tokenize words, split sentences to max_seq_len, and keep length
    # of each source word in tokens
    tokens, labels, lengths = tokenize_and_split_sentences(
        words, orig_labels, tokenizer, max_seq_len)

    # Extend each sentence to include context sentences
    combined_tokens, combined_labels, sentence_numbers, sentence_starts = combine_sentences2(
        tokens, labels, max_seq_len-1, seq_start)

    return Sentences(
        words, tokens, labels, lengths, combined_tokens, combined_labels, sentence_numbers, sentence_starts)


def read_data(input_file, tokenizer, max_seq_length):
    lines, tags, lengths = [], [], []

    def add_sentence(words, labels):
        split_tokens, split_labels, lens = tokenize_and_split(
            words, labels, tokenizer, max_seq_length-1)
        lines.extend(split_tokens)
        tags.extend(split_labels)
        lengths.extend(lens)

    curr_words, curr_labels = [], []
    with open(input_file) as rf:
        for line in rf:
            line = line.strip()
            if line:
                fields = line.split('\t')
                if len(fields) > 1:
                    curr_words.append(fields[0])
                    curr_labels.append(fields[1])
                else:
                    print('ignoring line: {}'.format(line), file=sys.stderr)
                    pass
            elif curr_words:
                # empty lines separate sentences
                add_sentence(curr_words, curr_labels)
                curr_words, curr_labels = [], []

        # Process last sentence also when there's no empty line after
        if curr_words:
            add_sentence(curr_words, curr_labels)
    return lines, tags, lengths


def write_result(fname, original, token_lengths, tokens, labels, predictions, mode='train'):
    lines=[]
    with open(fname,'w+') as f:
        toks = deque([val for sublist in tokens for val in sublist])
        labs = deque([val for sublist in labels for val in sublist])
        pred = deque([val for sublist in predictions for val in sublist])
        lengths = deque(token_lengths)
        sentences = []
        for sentence in original:
            sent = []
            for word in sentence:
                label = labs.popleft()
                predicted = pred.popleft()
                sent.append(predicted)
                for i in range(int(lengths.popleft())-1):
                    labs.popleft()
                    pred.popleft()                           
                if mode != 'predict':
                    line = "{}\t{}\t{}\n".format(word, label, predicted)
                else:
                    # In predict mode, labels are just placeholder dummies
                    line = "{}\t{}\n".format(word, predicted)
                f.write(line)
                lines.append(line)
            f.write("\n")
            sentences.append(sent)
    f.close()
    return lines, sentences


# Include maximum number of consecutive sentences to each sample
def combine_sentences(lines, tags, lengths, max_seq):
    lines_in_sample = []
    new_lines = []
    new_tags = []
    
    for i, line in enumerate(lines):
        line_numbers = [i]
        new_line = []
        new_line.extend(line)
        new_tag = []
        new_tag.extend(tags[i])
        j = 1
        linelen = len(lines[(i+j)%len(lines)])
        while (len(new_line) + linelen) < max_seq-2:
            new_line.append('[SEP]')
            new_tag.append('O')
            new_line.extend(lines[(i+j)%len(lines)])
            new_tag.extend(tags[(i+j)%len(tags)])
            line_numbers.append((i+j)%len(lines))
            j += 1
            linelen = len(lines[(i+j)%len(lines)])
        new_lines.append(new_line)
        new_tags.append(new_tag)
        lines_in_sample.append(line_numbers)
    return new_lines, new_tags, lines_in_sample

""" def combine_sentences2(lines, tags, max_seq, start=0):
    linelens = []
    new_lines = []
    new_tags = []
    
    for i, line in enumerate(lines):
        #linelens = [i]
        new_line = [0]*start
        new_line.extend(line)
        new_tag = [0]*start
        new_tag.extend(tags[i])
        j = 1
        linelen = len(lines[(i+j)%len(lines)])
        while (len(new_line) + linelen) < max_seq-2:
            new_line.append('[SEP]')
            new_tag.append('[SEP]')
            new_line.extend(lines[(i+j)%len(lines)])
            new_tag.extend(tags[(i+j)%len(tags)])
            #line_numbers.append((i+j)%len(lines))
            j += 1
            linelen = len(lines[(i+j)%len(lines)])

        #lines_in_sample.append(line_numbers)
        counter = start
        j=1
        ready = False
        while not ready:
            #print(i,j)
            prev_line = lines[i-j][:]
            prev_tags = tags[i-j][:]
            prev_line.append('[SEP]')
            prev_tags.append('[SEP]')
            #print(len(prev_line), len(prev_tags))
            if len(prev_line)< counter:
                new_line[(counter-len(prev_line)):counter]=prev_line
                new_tag[(counter-len(prev_line)):counter]=prev_tags
                counter -= len(prev_line)
                j+=1
            else:
                if counter > 2:
                    new_line[0:counter] = prev_line[-counter:]
                    new_tag[0:counter] = prev_tags[-counter:]
                    ready = True
                else:
                    new_line[0:counter] = ['[PAD]']*counter
                    new_tag[0:counter] = ['O']*counter
                    ready = True
        new_lines.append(new_line)
        new_tags.append(new_tag)           
    return new_lines, new_tags #, lines_in_sample """

def combine_sentences2(lines, tags, max_seq, start=0):
    lines_in_sample = []
    linestarts_in_sample = []
    new_lines = []
    new_tags = []
    position = start

    for i, line in enumerate(lines):
        line_starts = []
        line_numbers = []
        if start + len(line) < max_seq:                    
            new_line = [0]*start
            new_tag = [0]*start
            new_line.extend(line)
            new_tag.extend(tags[i])
            line_starts.append(start)
            line_numbers.append(i)
        else:
            position = max_seq - len(line) -1 
            new_line = [0]*position
            new_tag = [0]*position
            new_line.extend(line)
            new_tag.extend(tags[i])
            line_starts.append(position)
            line_numbers.append(i)
        j = 1
        next_idx = (i+j)%len(lines)
        ready = False
        while not ready:
            if len(lines[next_idx]) + len(new_line) < max_seq - 1: 
                new_line.append('[SEP]')
                new_tag.append('O')
                position = len(new_line)
                new_line.extend(lines[next_idx])
                new_tag.extend(tags[next_idx])
                line_starts.append(position)
                line_numbers.append(next_idx)
                j += 1
                next_idx = (i+j)%len(lines)
            else:
                new_line.append('[SEP]')
                new_tag.append('O')
                position = len(new_line)
                new_line.extend(lines[next_idx][0:(max_seq-position)])
                new_tag.extend(tags[next_idx][0:(max_seq-position)])
                ready = True                

        #lines_in_sample.append(line_numbers)

        j=1
        ready = False
        while not ready:
            counter = line_starts[0]
            #print(counter)
            prev_line = lines[i-j][:]
            prev_tags = tags[i-j][:]
            prev_line.append('[SEP]')
            prev_tags.append('O')
            #print(len(prev_line), len(prev_tags))
            if len(prev_line)<= counter:
                new_line[(counter-len(prev_line)):counter]=prev_line
                new_tag[(counter-len(prev_line)):counter]=prev_tags
                line_starts.insert(0,counter-len(prev_line))
                line_numbers.insert(0,i-j)  #negative numbers are indices to end of lines array
                j+=1
            else:
                if counter > 2:
                    new_line[0:counter] = prev_line[-counter:]
                    new_tag[0:counter] = prev_tags[-counter:]
                    ready = True
                else:
                    new_line[0:counter] = ['[PAD]']*counter
                    new_tag[0:counter] = ['O']*counter
                    ready = True
        new_lines.append(new_line)
        new_tags.append(new_tag)
        lines_in_sample.append(line_numbers)
        linestarts_in_sample.append(line_starts)
    return new_lines, new_tags, lines_in_sample, linestarts_in_sample

def get_predictions(predicted, lines, line_numbers):
    first_pred = []
    final_pred = []
    predictions = [[] for _ in range(len(lines))]
    for i, sample in enumerate(predicted):
        idx = 1
        for j, line_number in enumerate(line_numbers[i]):
            predictions[line_number].append(sample[idx:idx+len(lines[line_number])])
            if j == 0:
                first_pred.append(sample[idx:idx+len(lines[line_number])])
            idx+=len(lines[line_number])+1
    for i, prediction in enumerate(predictions):
        pred = []
        arr = np.stack(prediction, axis=0)
        for j in arr.T:
            u,c = np.unique(j, return_counts=True)
            pred.append(u[np.argmax(c)])
        final_pred.append(pred)
    return final_pred, first_pred

def get_predictions2(probs, lines, line_numbers):
    first_pred = []
    final_pred = []
    predictions = []
    p_first = []
    for i, line in enumerate(lines):
        predictions.append(np.zeros((len(line),probs.shape[-1])))  #create empty array for each line
    
    for i, sample in enumerate(probs):
        idx = 1
        for j, line_number in enumerate(line_numbers[i]):
            if j == 0:
                p_first.append(sample[idx:idx+len(lines[line_number]),:])
            predictions[line_number] += sample[idx:idx+len(lines[line_number]),:]
            idx+=len(lines[line_number])+1
    
    for k, line in enumerate(predictions): 
        final_pred.append(np.argmax(line, axis=-1))
        first_pred.append(np.argmax(p_first[k],axis=-1))
            
    return final_pred, first_pred

def process_docs(docs, doc_tags, line_ids, tokenizer, seq_len):
    
    f_words = []
    f_tokens = []
    f_labels = []
    f_lengths = []
    f_combined_tokens = []
    f_combined_labels = []
    f_sentence_numbers = []
    f_sentence_starts = []
    start_sentence_number = 0
    for i, doc in enumerate(docs):
            tokens, labels, lengths = tokenize_and_split_sentences(
                doc, doc_tags[i], tokenizer, seq_len)
            combined_tokens, combined_labels, sentence_numbers = combine_sentences(tokens, 
                                                                                   labels,
                                                                                   lengths,
                                                                                   seq_len)
            for numbers in sentence_numbers:
                f_sentence_numbers.append([num+start_sentence_number for num in numbers])
            start_sentence_number += len(tokens)
            f_words.extend(doc)
            f_tokens.extend(tokens)
            f_labels.extend(labels)
            f_lengths.extend(lengths)
            f_combined_tokens.extend(combined_tokens)
            f_combined_labels.extend(combined_labels)
            f_sentence_starts.extend([0]*len(tokens))
            
    return Sentences(f_words, f_tokens, f_labels, f_lengths, f_combined_tokens, f_combined_labels, f_sentence_numbers, f_sentence_starts)         
                
def split_to_documents(sentences, tags):
    documents = []
    documents_tags = []
    doc_idx = 0
    document = []
    d_tags = []
    line_ids =[[]]
    for i, sentence in enumerate(sentences):
        if sentence[0].startswith("-DOCSTART-") and i!=0:
            documents.append(document)
            documents_tags.append(d_tags)
            document = []
            d_tags = []
            line_ids.append([])
            doc_idx+=1
        document.append(sentence)
        d_tags.append(tags[i])
        line_ids[doc_idx].append(i)
    if documents:
            documents.append(document)
            documents_tags.append(d_tags)
    return documents, documents_tags, line_ids
            
def process_no_context(train_words, train_tags, tokenizer, seq_len):
    train_tokens, train_labels, train_lengths = tokenize_and_split_sentences(train_words, train_tags, tokenizer, seq_len)
    sentence_numbers = []
    sentence_starts = []
    for i, line in enumerate(train_tokens):
        sentence_numbers.append([i])
        sentence_starts.append([0])
    return Sentences(train_words, train_tokens, train_labels, train_lengths, train_tokens, train_labels, sentence_numbers, sentence_starts)
