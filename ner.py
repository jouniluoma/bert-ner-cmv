import os
os.environ['TF_KERAS'] = '1'

import sys
import numpy as np
import conlleval
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import multi_gpu_model

from common import encode, label_encode, write_result
from common import load_pretrained
from common import create_ner_model, create_optimizer, argument_parser
from common import read_conll, process_sentences, get_labels
from common import save_ner_model
from common import load_ner_model
from common import get_predictions, combine_sentences2
from common import Sentences
from common import tokenize_and_split_sentences, split_to_documents, get_predictions2, process_docs, process_no_context




def main(argv):

    argparser = argument_parser()
    args = argparser.parse_args(argv[1:])
    seq_len = args.max_seq_length    # abbreviation

    pretrained_model, tokenizer = load_pretrained(args)

    train_words, train_tags = read_conll(args.train_data)
    test_words, test_tags = read_conll(args.test_data)


    print(args.no_context)

    if args.no_context:
        train_data = process_no_context(train_words, train_tags, tokenizer, seq_len)
        test_data = process_no_context(test_words, test_tags, tokenizer, seq_len)
    elif args.documentwise:
        tr_docs, tr_doc_tags, tr_line_ids = split_to_documents(train_words, train_tags)
        te_docs, te_doc_tags, te_line_ids = split_to_documents(test_words, test_tags)
        train_data = process_docs(tr_docs, tr_doc_tags, tr_line_ids, tokenizer, seq_len)
        test_data = process_docs(te_docs, te_doc_tags, te_line_ids, tokenizer, seq_len)
    else:
        train_data = process_sentences(train_words, train_tags, tokenizer, seq_len, args.predict_position)
        test_data = process_sentences(test_words, test_tags, tokenizer, seq_len, args.predict_position)
    
    label_list = get_labels(train_data.labels)
    tag_map = { l: i for i, l in enumerate(label_list) }
    inv_tag_map = { v: k for k, v in tag_map.items() }

    train_x = encode(train_data.combined_tokens, tokenizer, seq_len)
    test_x = encode(test_data.combined_tokens, tokenizer, seq_len)
    train_y, train_weights = label_encode(train_data.combined_labels, tag_map, seq_len)
    test_y, test_weights = label_encode(test_data.combined_labels, tag_map, seq_len)


    if args.use_ner_model and (args.ner_model_dir is not None):
        ner_model, tokenizer, labels, config = load_ner_model(args.ner_model_dir)
    else:
        optimizer = create_optimizer(len(train_x[0]), args)
        model = create_ner_model(pretrained_model, len(tag_map))
        if args.num_gpus > 1:
            ner_model = multi_gpu_model(model, args.num_gpus)
        else:
            ner_model = model

        ner_model.compile(
            optimizer,
            loss='sparse_categorical_crossentropy',
            sample_weight_mode='temporal',
            metrics=['sparse_categorical_accuracy']
            )
                
        ner_model.fit(
            train_x,
            train_y,
            sample_weight=train_weights,
            epochs=args.num_train_epochs,
            batch_size=args.batch_size
            )
        if args.ner_model_dir is not None:
            label_list = [v for k, v in sorted(list(inv_tag_map.items()))]
            save_ner_model(ner_model, tokenizer, label_list, args)

    
    probs = ner_model.predict(test_x, batch_size=args.batch_size)
    preds = np.argmax(probs, axis=-1)
    
    results = []
    m_names = []
    if args.no_context:
        pr_ensemble, pr_test_first = get_predictions(preds, test_data.tokens, test_data.sentence_numbers)
        output_file = "output/{}-NC.tsv".format(args.output_file)
        m_names.append('NC')  
        ensemble = []
        for i,pred in enumerate(pr_test_first):
            ensemble.append([inv_tag_map[t] for t in pred])
        lines_ensemble, sentences_ensemble = write_result(
            output_file, test_data.words, test_data.lengths,
            test_data.tokens, test_data.labels, ensemble
            )
        c = conlleval.evaluate(lines_ensemble)
        conlleval.report(c)
        results.append([conlleval.metrics(c)[0].prec, conlleval.metrics(c)[0].rec, conlleval.metrics(c)[0].fscore])



    else:
        # First tag then vote
        pr_ensemble, pr_test_first = get_predictions(preds, test_data.tokens, test_data.sentence_numbers)
        # Accumulate probabilities, then vote
        prob_ensemble, prob_test_first = get_predictions2(probs, test_data.tokens, test_data.sentence_numbers)
        ens = [pr_ensemble, prob_ensemble, pr_test_first, prob_test_first]
        if args.documentwise:
            # D-CMV: Documentwise CMV
            # D-CMVP: Documetwise CMV, probs summed, argmax after that
            # D-F: Documentwise First
            # D-FP: Same as D-FP 
            method_names = ['D-CMV','D-CMVP','D-F','D-FP']  
        else:           
            method_names = ['CMV','CMVP','F','FP']
        for i, ensem in enumerate(ens):
            ensemble = []
            for j,pred in enumerate(ensem):
                ensemble.append([inv_tag_map[t] for t in pred])
            output_file = "output/{}-{}.tsv".format(args.output_file, method_names[i])
            lines_ensemble, sentences_ensemble = write_result(
                    output_file, test_data.words, test_data.lengths,
                    test_data.tokens, test_data.labels, ensemble)
            print("Model trained: ", args.ner_model_dir)
            print("Seq-len: ", args.max_seq_length)
            print("Learning rate: ", args.learning_rate)
            print("Batch Size: ", args.batch_size)
            print("Epochs: ", args.num_train_epochs)
            print("Training data: ", args.train_data)
            print("Testing data: ", args.test_data)
            print("")
            print("Results with {}".format(method_names[i]))
            c = conlleval.evaluate(lines_ensemble)
            print("")
            conlleval.report(c)
            results.append([conlleval.metrics(c)[0].prec, conlleval.metrics(c)[0].rec, conlleval.metrics(c)[0].fscore])
            m_names.extedm(method_names)

        
    if args.sentence_in_context:     
        starting_pos = np.arange(0,seq_len+1,32)
        starting_pos[0] = 1
        m_names.extend(starting_pos)
        for start_p in starting_pos:
            tt_lines, tt_tags, line_nos, line_starts = combine_sentences2(test_data.tokens, test_data.labels, seq_len-1, start_p-1)
            tt_x = encode(tt_lines, tokenizer, seq_len)
            tt_y, train_weights = label_encode(tt_tags, tag_map, seq_len)
            probs = ner_model.predict(tt_x, batch_size=args.batch_size)
            preds = np.argmax(probs, axis=-1)


            pred_tags = []
            for i, pred in enumerate(preds):
                idx = line_nos[i].index(i)
                pred_tags.append([inv_tag_map[t] for t in pred[line_starts[i][idx]+1:line_starts[i][idx]+len(test_data.tokens[i])+1]])
                
            output_file = "output/{}-{}.tsv".format(args.output_file, start_p)
            lines_first, sentences_first = write_result(
                output_file, test_data.words, test_data.lengths,
                test_data.tokens, test_data.labels, pred_tags
            )
            print("")
            print("Results with prediction starting position ", start_p)
            c = conlleval.evaluate(lines_first)
            conlleval.report(c)
            results.append([conlleval.metrics(c)[0].prec, conlleval.metrics(c)[0].rec, conlleval.metrics(c)[0].fscore])

    result_file = "./results/results-{}.csv".format(args.output_file) 
    with open(result_file, 'w+') as f:
        for i, line in enumerate(results):
            params = "{},{},{},{},{},{},{},{},{}".format(args.output_file,
                                            args.max_seq_length, 
                                            args.bert_config_file, 
                                            args.num_train_epochs, 
                                            args.learning_rate,
                                            args.batch_size,
                                            args.predict_position,
                                            args.train_data,
                                            args.test_data)
            f.write(params)
            f.write(",{}".format(m_names[i]))
            for item in line:
                f.write(",{}".format(item))
            f.write('\n') 

    for i in results:
        print(i)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
