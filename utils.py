from datasets import load_dataset
import pandas as pd
import torch
from transformers import DistilBertTokenizerFast, DistilBertForQuestionAnswering, pipeline
import json

#torch dataset class for loading datasets
class SquadDS(torch.utils.data.Dataset):

    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

#Unpacking the squad dataset into questions, answers and context for each split
def unpack_dataset():
    train_ds = load_dataset('squad', split='train')
    test_ds = load_dataset('squad', split='validation')
    df1 = pd.DataFrame(train_ds)
    df2 = pd.DataFrame(test_ds)

    #The following indices have been throwing errors when loaded into the dataset. Thus, we've removed them from training
    faulty_df1 = [3275, 3276, 11308, 49094, 50763, 57586, 57587, 57588, 57589, 57590, 60024, 60027, 66282]

    faulty_df2 = [4145, 4146, 4264, 4269, 4282, 4283, 4851, 4852, 4853]

    df1 = df1.drop(faulty_df1)
    df1 = df1.reset_index(drop=True)
    df2 = df2.drop(faulty_df2)
    df2 = df2.reset_index(drop=True)

    train_questions = df1.question.tolist()
    test_questions = df2.question.tolist()

    train_context = df1.context.tolist()
    test_context = df2.context.tolist()

    train_answers = df1.answers.tolist()
    test_answers = df2.answers.tolist()

    return train_questions, train_context, train_answers, test_questions, test_context, test_answers

#add answer endings to the set
def add_answer_end(answers, context):
    for answer, context in zip(answers, context):
        if answer['answer_start'] is None:
            answer['answer_end'] = None
        else:
            if len(answer['text']) > 1:
                temp_end = []
                temp_start = []
                for ans, start in zip(answer['text'], answer['answer_start']):
                    gold = ans
                    end_idx = start + len(gold)
                    if context[start:end_idx] == gold:
                        temp_start.append(start)
                        temp_end.append(end_idx)

                    elif context[start - 1:end_idx - 1] == gold:
                        temp_start.append(start - 1)
                        temp_end.append(end_idx - 1)

                    elif context[start - 2:end_idx - 2] == gold:
                        temp_start.append(start - 2)
                        temp_end.append(end_idx - 2)

                answer['answer_end'] = temp_end
                answer['answer_start'] = temp_start

            else:
                gold = answer['text'][0]
                start_idx = answer['answer_start']
                end_idx = start_idx[0] + len(gold)
                if context[start_idx[0]:end_idx] == gold:
                    answer['answer_end'] = [end_idx]
                elif context[start_idx[0] - 1:end_idx - 1] == gold:
                    answer['answer_start'] = [start_idx[0] - 1]
                    answer['answer_end'] = [end_idx - 1]  # When the gold label is off by one character
                elif context[start_idx[0] - 2:end_idx - 2] == gold:
                    answer['answer_start'] = [start_idx[0] - 2]
                    answer['answer_end'] = [end_idx - 2]  # When the gold label is off by two characters

#ensure that start/end positions match token positions
def define_token_position(encoding, answers):
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased-distilled-squad')
    start_positions = []
    end_positions = []

    for idx in range(len(answers)):
        if len(answers[idx]['answer_start']) > 1:
            temp_st = []
            temp_en = []
            for ans_st in answers[idx]['answer_start']:
                if ans_st is None:
                    temp_st.append(tokenizer.model_max_length - 1)
                else:
                    temp_st.append(encoding.char_to_token(idx, ans_st))
            for ans_en in answers[idx]['answer_end']:
                if ans_en is None:
                    temp_en.append(tokenizer.model_max_length - 1)
                else:
                    temp_en.append(encoding.char_to_token(idx, ans_en - 1))
            start_positions.append(temp_st)
            end_positions.append(temp_en)


        else:

            if answers[idx]['answer_start'] is None:

                st = tokenizer.model_max_length - 1
            else:

                st = encoding.char_to_token(idx, answers[idx]['answer_start'][0])
            if answers[idx]['answer_end'] is None:

                en = tokenizer.model_max_length - 1
            else:
                # answers[idx]['answer_end'] = encoding.char_to_token(idx, answers[idx]['answer_end'][0] - 1)
                en = encoding.char_to_token(idx, answers[idx]['answer_end'][0] - 1)
            # if None, the answer passage has been truncated due to words > 512 so setting last position as 511
            start_positions.append(st)
            end_positions.append(en)

    encoding.update({'start_positions': start_positions, 'end_positions': end_positions})

#save metrics in a json file
def save_metrics_json(loss_epoch,
                      acc_epoch,
                      f1_epoch,
                      test_acc,
                      avg_test_f1,
                      file_path):
    metrics = {}
    metrics["train"] = {}
    metrics["test"] = {}

    for i in range(len(loss_epoch)):
        metrics["train"][f"epoch{i + 1}"] = {}
        metrics["train"][f"epoch{i + 1}"]["loss"] = loss_epoch[i]
        metrics["train"][f"epoch{i + 1}"]["accuracy"] = acc_epoch[i]
        metrics["train"][f"epoch{i + 1}"]["f1_score"] = f1_epoch[i]
        metrics["test"]["accuracy"] = test_acc
        metrics["test"]["f1_score"] = avg_test_f1
    with open(file_path, 'w') as json_file:
        json.dump(metrics, json_file, indent=4)


def load_model(path):
    """
    Load and return a DistilBertForQuestionAnswering model with parameters from the file provided.

    Parameters
    ----------
    path : path to the saved model parameters.

    Returns
    -------
    model : a DistilBertForQuestionAnswering model with the parameters from the file provided.

    """
    state_dict = torch.load(path, map_location=torch.device('cuda:2'))

    model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased", state_dict=state_dict)
    model.eval()
    return model

#Shap training function
def f(questions, start):
    outs = []
    for q in questions:
        question, context = q.split("[SEP]")
        d = pmodel.tokenizer(question, context)
        out = pmodel.model.forward(**{k: torch.tensor(d[k]).reshape(1, -1) for k in d})
        logits = out.start_logits if start else out.end_logits
        outs.append(logits.reshape(-1).detach().numpy())
    return outs

#loads model for auxiliary shap functions
model = load_model('distillbert_done')
# define two predictions, one that outputs the logits for the range start,
pmodel = pipeline(task='question-answering', model=model,
                  tokenizer=DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased-distilled-squad'))

def f_start(questions):
    return f(questions, True)


def f_end(questions):
    return f(questions, False)


def out_names(inputs):
    question, context = inputs.split("[SEP]")
    d = pmodel.tokenizer(question, context)
    return [pmodel.tokenizer.decode([id]) for id in d["input_ids"]]

#get's the output of true and false items for shap visualizations
def shap_test(tt):
    device = 'cuda:3'  # if torch.cuda.is_available()# else 'cpu'

    # print('Cached:   ', round(torch.cuda.memory_reserved(torch.cuda.current_device)/1024**3,1), 'GB')
    # print('Allocated:', round(torch.cuda.memory_allocated(torch.cuda.current_device)/1024**3,1), 'GB')

    torch.cuda.empty_cache()  # Free up GPU memory

    true = []
    false = []
    model.to(device)
    device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
    input_ids = tt['input_ids'].to(device)
    attention_mask = tt['attention_mask'].to(device)
    start_positions = tt['start_positions'].to(device)
    end_positions = tt['end_positions'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    logits = outputs.start_logits
    pred = torch.argmax(logits, dim=1)

    # print(start_positions.shape[0])
    if start_positions.numel() > 1:
        for j in range(start_positions.shape[0]):
            if pred == start_positions[j]:
                return True

        else:
            return False
    else:
        return False
