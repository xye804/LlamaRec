import numpy as np
import torch

from torch.utils.data import DataLoader
from unsloth import FastLanguageModel
from dataset import LlamaRecDataset
from parameters import parse_args
from preprocess import DataPreprocess

args = parse_args()


def load_llm():
    llama, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Meta-Llama-3.1-8B-bnb-4bit",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
        token="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",
    )

    for param in llama.model.parameters():
        param.requires_grad = False

    for layer in llama.model.layers[-args.num_non_frozen_layers:]:
        for param in layer.parameters():
            if param.dtype in [torch.float32, torch.float64, torch.float16]:
                param.requires_grad = True

    return llama, tokenizer


def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)


def mrr(y_true, y_pred):
    mrrs = []
    for true, scores in zip(y_true, y_pred):
        sorted_indices = np.argsort(scores)[::-1]
        for rank, index in enumerate(sorted_indices, start=1):
            if true[index] == 1:
                mrrs.append(1 / rank)
                break
        else:
            mrrs.append(0)
    return np.mean(mrrs)


def collate_fn(batch):
    return zip(*batch)


def prepare_data():
    train_news_df, train_behaviors = DataPreprocess("train").process()
    news_df, dev_behaviors, test_behaviors = DataPreprocess("dev").process()

    train_dataset = LlamaRecDataset(train_news_df, train_behaviors, args)
    val_dataset = LlamaRecDataset(news_df, dev_behaviors, args, True)
    test_dataset = LlamaRecDataset(news_df, test_behaviors, args, True)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn,
                                  pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                                collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                                collate_fn=collate_fn)                            

    return train_dataloader, val_dataloader, test_dataloader
