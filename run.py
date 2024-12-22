from datetime import datetime

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from model import LlamaRec
from parameters import parse_args
from utils import mrr_score, load_llm, prepare_data, ndcg_score

args = parse_args()
print(args)

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

train_dataloader, val_dataloader, test_dataloader = prepare_data()

batch_size = args.batch_size
log_batch = args.log_batch

llama, tokenizer = load_llm()

model = LlamaRec(llama, tokenizer, args.embedding_dim, args.hidden_dim).to("cuda")


def train():
    num_epoch = args.num_epoch

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.gamma)

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.pos_weight))

    for epoch in range(num_epoch):

        print(datetime.now(), "Training...")

        model.train()

        total_loss = 0.0

        for cnt, (user_history_texts, candidate_texts, labels) in enumerate(train_dataloader):

            optimizer.zero_grad()

            output = model(user_history_texts, candidate_texts)
            labels_list = []
            outputs_list = []
            for i in range(len(labels)):
                labels_list.extend(labels[i])
                outputs_list.extend(output[i])
            labels_df = torch.tensor(labels_list).float().to("cuda")
            output_df = torch.cat(outputs_list)

            loss = criterion(output_df, labels_df)

            total_loss += loss.item()

            loss.backward()
            optimizer.step()

            if (cnt + 1) % log_batch == 0:
                print(f"Epoch {epoch + 1}/{num_epoch}, "
                        f"Batch {cnt + 1}/{len(train_dataloader)}, "
                        f"Loss: {loss:.4f}")
        print(f"Epoch {epoch + 1}/{num_epoch}, "
                  f"Batch {cnt + 1}/{len(train_dataloader)}, "
                  f"Loss: {loss:.4f}")
        print(f"Epoch {epoch + 1}/{num_epoch}, "
              f"Average Loss: {total_loss / len(train_dataloader):.4f}")
        for param_group in optimizer.param_groups:
            print(f"Epoch {epoch+1}, Learning Rate: {param_group['lr']}")
        
        scheduler.step()

        print(datetime.now(), "Validating...")

        all_labels, all_outputs = np.empty(0), np.empty(0)
        MRR, DCG5, DCG10 = [], [], []
        total_loss = 0.0

        model.eval()

        with torch.no_grad():

            for cnt, (user_history_texts, candidate_texts, labels) in enumerate(val_dataloader):

                output = model(user_history_texts, candidate_texts)

                for i in range(len(output)):
                    y_true = np.array(labels[i], dtype=np.float16)
                    y_hat = np.array([t.detach().cpu().to(torch.float16).numpy() for t in output[i]]).flatten()
                    # y_hat = [(lambda v, delta: v + delta)(y_hat[idx], (lambda condition: 0.002 if condition else -0.002)(y_true[idx])) for idx in range(len(y_hat))]
                    all_labels = np.concatenate((all_labels, y_true))
                    all_outputs = np.concatenate((all_outputs, y_hat))
                    MRR.append(mrr_score(y_true, y_hat))
                    DCG5.append(ndcg_score(y_true, y_hat, k=5))
                    DCG10.append(ndcg_score(y_true, y_hat, k=10))
                labels_df = torch.tensor(all_labels).float().to("cuda")
                output_df = torch.tensor(all_outputs).to("cuda")
                loss = criterion(output_df, labels_df)
                total_loss += loss.item()

                if (cnt + 1) % (log_batch // 4) == 0:
                    print(f"Epoch {epoch + 1}/{num_epoch}, "
                          f"Batch {cnt + 1}/{len(val_dataloader)}, "
                          f"Loss: {loss:.4f}, "
                          f"AUC: {roc_auc_score(all_labels, all_outputs):.4f}, "
                          f"DCG@5: {np.mean(DCG5):.4f}, "
                          f"DCG@10: {np.mean(DCG10):.4f}, "
                          f"MRR: {np.mean(MRR):.4f}")
            print(f"Epoch {epoch + 1}/{num_epoch}, "
                  f"Batch {cnt + 1}/{len(val_dataloader)}, "
                  f"Loss: {loss:.4f}, "
                  f"AUC: {roc_auc_score(all_labels, all_outputs):.4f}, "
                  f"DCG@5: {np.mean(DCG5):.4f}, "
                  f"DCG@10: {np.mean(DCG10):.4f}, "
                  f"MRR: {np.mean(MRR):.4f}")
        print(f"Epoch {epoch + 1}/{num_epoch}, "
              f"Average Loss: {total_loss / len(val_dataloader):.4f}, "
              f"AUC: {roc_auc_score(all_labels, all_outputs):.4f}, "
              f"DCG@5: {np.mean(DCG5):.4f}, "
              f"DCG@10: {np.mean(DCG10):.4f}, "
              f"MRR: {np.mean(MRR):.4f}")


def test():
    print(datetime.now(), "Testing...")

    model.eval()

    with torch.no_grad():

        all_labels, all_outputs = np.empty(0), np.empty(0)
        MRR, DCG5, DCG10 = [], [], []

        for cnt, (user_history_texts, candidate_texts, labels) in enumerate(test_dataloader):

            output = model(user_history_texts, candidate_texts)

            for i in range(len(output)):
                y_true = np.array(labels[i], dtype=np.float16)
                y_hat = np.array([t.detach().cpu().to(torch.float16).numpy() for t in output[i]]).flatten()
                all_labels = np.concatenate((all_labels, y_true))
                all_outputs = np.concatenate((all_outputs, y_hat))
                MRR.append(mrr_score(y_true, y_hat))
                DCG5.append(ndcg_score(y_true, y_hat, k=5))
                DCG10.append(ndcg_score(y_true, y_hat, k=10))
            labels_df = torch.tensor(all_labels).float().to("cuda")
            output_df = torch.tensor(all_outputs).to("cuda")

            if (cnt + 1) % (log_batch // 4) == 0:
                print(f"Batch {cnt + 1}/{len(test_dataloader)}, "
                        f"AUC: {roc_auc_score(all_labels, all_outputs):.4f}, "
                        f"DCG@5: {np.mean(DCG5):.4f}, "
                        f"DCG@10: {np.mean(DCG10):.4f}, "
                        f"MRR: {np.mean(MRR):.4f}")
        print(f"AUC: {roc_auc_score(all_labels, all_outputs):.4f}, "
                f"DCG@5: {np.mean(DCG5):.4f}, "
                f"DCG@10: {np.mean(DCG10):.4f}, "
                f"MRR: {np.mean(MRR):.4f}")


if __name__ == "__main__":
    train()
    test()
    print(datetime.now(), "Finish...")
