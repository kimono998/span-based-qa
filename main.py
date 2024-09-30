import argparse 
import random
import torch
import numpy as np
from transformers import DistilBertTokenizerFast,DistilBertForQuestionAnswering,pipeline

import shap

from utils import unpack_dataset, SquadDS, add_answer_end, define_token_position, load_model, f, f_end, f_start, \
    out_names, save_metrics_json, shap_test
from model import train, test


def main():
    parser = argparse.ArgumentParser(
        description='SQUAD Question Answering Dataset'
    )

    parser.add_argument(
        "--load", dest="load",
        help="Loads the data",
        action="store_true",
        default=None

    )

    parser.add_argument(
        "--Train", dest="Train",
        help="trains the dataset using Transformers(DistilBertForQuestionAnswering)",
        action="store_true"
    )


    parser.add_argument(
        "--test", dest="test",
        help="Runs the model on the test_set",
        action="store_true"
    )

    parser.add_argument(
        "--shap_eval", dest="shap_eval",
        help="shows feature importance",
        action="store_true"
    )


    args = parser.parse_args()
    
    if args.load:
        print('Loading dataset...')
        train_questions, train_context, train_answers, test_questions, test_context, test_answers = unpack_dataset()
        print('Done')
        print('Adding answer endings...')
        add_answer_end(train_answers, train_context)
        add_answer_end(test_answers, test_context)
        print('Done!')
        print('Extracting encodings...')
        train_encodings = tokenizer(train_context, train_questions, truncation=True, padding=True)
        test_encodings = tokenizer(test_context, test_questions, truncation=True, padding=True)
        print('Done')
        print('Defining token positions...')
        define_token_position(train_encodings, train_answers)
        define_token_position(test_encodings, test_answers)
        
        print(f'Everything loaded!')
        
    
    if args.Train:
        print('Training initiated')
        torch.cuda.empty_cache()  # Free up GPU memory

        model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")
        tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased-distilled-squad')

        train_questions, train_context, train_answers, test_questions, test_context, test_answers = unpack_dataset()
        add_answer_end(train_answers, train_context)
        add_answer_end(test_answers, test_context)

        train_encodings = tokenizer(train_context, train_questions, truncation=True, padding=True)
        test_encodings = tokenizer(test_context, test_questions, truncation=True, padding=True)

        define_token_position(train_encodings, train_answers)
        define_token_position(test_encodings, test_answers)

        # -----------#continue training#------------
        #saved_state_dict = torch.load('model/distillbert_done')
        #model.load_state_dict(saved_state_dict)
        # -----------###################------------

        train_dataset = SquadDS(train_encodings)
        test_dataset = SquadDS(test_encodings)
        print('Dataset loaded!')
        print('Now training...')
        model, loss_epoch, f1_epoch, acc_epoch = train(model, train_dataset)
        avg_test_f1, test_acc = test(model, test_dataset)
        save_metrics_json(loss_epoch, acc_epoch, f1_epoch, test_acc, avg_test_f1, 'metrics_distillbert.json')
        print('Everything done.')
    if args.test:


        print('Testing')
        model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")
        tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased-distilled-squad')


        # --------/Load the model/--------#
        #state_dict = torch.load('model/distillbert_done')
        #model.load_state_dict(state_dict)
        # --------/-------------/--------#

        _, _, _, test_questions, test_context, test_answers = unpack_dataset()
        add_answer_end(test_answers, test_context)
        test_encodings = tokenizer(test_context, test_questions, truncation=True, padding=True)
        define_token_position(test_encodings, test_answers)
        test_dataset = SquadDS(test_encodings)
        avg_test_f1, test_acc = test(model, test_dataset)

    if args.shap_eval:

        print('SHAP visualizations')
        model = load_model('model/distillbert_done')
        # define two predictions, one that outputs the logits for the range start,
        pmodel = pipeline(task='question-answering', model=model,
                          tokenizer=DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased-distilled-squad'))

        # attach a dynamic output_names property to the models so we can plot the tokens at each output position

        f_start.output_names = out_names
        f_end.output_names = out_names

        true_examples = []
        false_examples = []

        for idx, item in enumerate(test_dataset):
            out = shap_test(item)

            if out == True:
                true_examples.append(idx)
            if out == False:
                false_examples.append(idx)

        true_ex = []
        for idx in true_examples:
            true_ex.append('[SEP]'.join((test_questions[idx], test_context[idx])))

        false_ex = []
        for idx in false_examples:
            false_ex.append('[SEP]'.join((test_questions[idx], test_context[idx])))

        rand_true = random.choices(true_ex, k=10)
        rand_false = random.choices(false_ex, k=10)

        explainer_start = shap.Explainer(f_start, pmodel.tokenizer)
        shap_values_true_start = explainer_start(rand_true[0])

        shap.plots.text(shap_values_true_start)

if __name__ == "__main__":
    main()

