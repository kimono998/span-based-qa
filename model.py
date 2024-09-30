import torch
import tqdm

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import AdamW
from sklearn.metrics import f1_score
from tqdm import tqdm

#Model save path
PATH = 'model/distillbert_done'

#Training function. Arguments: Model and the Dataset
def train(model, train_dataset):
    #select device (cuda)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #save checkpoints
    save_path = 'model_checkpoint_epoch{}.pt'
    save_interval = 1

    #load the dataset in the dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    loss_epoch = []
    acc_epoch = []
    f1_epoch = []
    model.to(device)

    #run training for 6 epochs
    for epoch in range(6):

        print('Cached:   ', round(torch.cuda.memory_reserved(torch.cuda.current_device) / 1024 ** 3, 1), 'GB')
        print('Allocated:', round(torch.cuda.memory_allocated(torch.cuda.current_device) / 1024 ** 3, 1), 'GB')

        total_loss = 0
        total_correct = 0
        total_pred = 0
        true_labels = []
        predicted_labels = []
        model.train()

        for batch in tqdm(train_dataloader):
            
            torch.cuda.empty_cache()  # Free up GPU memory

            optimizer.zero_grad()
            #move all of the relevant attributes to gpu
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)

            #get model outputs
            outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions,
                            end_positions=end_positions)

            #get start_logits
            logits = outputs.start_logits
            pred = torch.argmax(logits, dim=1)
            total_correct += torch.sum(pred == start_positions).item()
            total_pred += len(start_positions)

            loss = outputs[0]
            total_loss += loss.item()

            true_labels.extend(start_positions.tolist())
            predicted_labels.extend(pred.tolist())
            avg_train_f1 = f1_score(true_labels, predicted_labels, average='macro')

            loss.backward()
            optimizer.step()

        avg_train_loss = total_loss / len(train_dataloader)
        avg_train_acc = total_correct / total_pred
        acc_epoch.append(avg_train_acc)
        loss_epoch.append(avg_train_loss)
        f1_epoch.append(avg_train_f1)
        print("average training loss: {0:.2f}".format(avg_train_loss))
        print("average training accuracy: {0:.2f}".format(avg_train_acc))
        print("average training f1_score: {0:.2f}".format(avg_train_f1))

        if (epoch + 1) % save_interval == 0:
            torch.save(model.state_dict(), save_path.format(epoch + 1))

    torch.save(model.state_dict(), PATH)

    return model, loss_epoch, f1_epoch, acc_epoch

#custom collate function to fix padding errors during testing
def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    start_positions = [item['start_positions'] for item in batch]
    end_positions = [item['end_positions'] for item in batch]

    input_ids_padded = pad_sequence([x.unsqueeze(0) if x.ndim == 0 else x for x in input_ids], batch_first=True,
                                    padding_value=0)
    attention_mask_padded = pad_sequence([x.unsqueeze(0) if x.ndim == 0 else x for x in attention_mask],
                                         batch_first=True, padding_value=0)
    start_positions_padded = pad_sequence([x.unsqueeze(0) if x.ndim == 0 else x for x in start_positions],
                                          batch_first=True, padding_value=-1)
    end_positions_padded = pad_sequence([x.unsqueeze(0) if x.ndim == 0 else x for x in end_positions], batch_first=True,
                                        padding_value=-1)

    return {
        'input_ids': input_ids_padded,
        'attention_mask': attention_mask_padded,
        'start_positions': start_positions_padded,
        'end_positions': end_positions_padded
    }

#testing function. Predicts the start of the answer in the test dataset
def test(model, test_dataset):
    device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    total_correct = 0
    total_pred = 0
    predicted_labels = []
    true_labels = []
    model.eval()
    model.to(device)

    for batch in tqdm(test_dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)

        logits = outputs.start_logits
        pred = torch.argmax(logits, dim=1)
        for i in range(len(pred)):
            for j in range(len(start_positions[i])):
                if pred[i] == start_positions[i][j]:
                    total_correct += 1
                    break

        total_pred += len(start_positions)

        true_labels.extend(start_positions[:, 2].tolist())
        predicted_labels.extend(pred.tolist())
        avg_test_f1 = f1_score(true_labels, predicted_labels, average='macro')

    test_acc = total_correct / total_pred
    print("average testing accuracy: {0:.2f}".format(test_acc))
    print("average testing f1_score: {0:.2f}".format(avg_test_f1))

    return avg_test_f1, test_acc
