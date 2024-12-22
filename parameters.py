import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=int, default=3e-5)
    parser.add_argument("--gamma", type=int, default=0.2)
    parser.add_argument("--pos_weight", type=int, default=4)
    parser.add_argument("--np_ratio", type=int, default=4)
    parser.add_argument("--num_epoch", type=int, default=3)
    parser.add_argument("--embedding_dim", type=int, default=4096)
    parser.add_argument("--hidden_dim", type=int, default=2048)
    parser.add_argument("--num_attention_heads", type=int, default=8)
    parser.add_argument("--num_non_frozen_layers", type=int, default=2)
    parser.add_argument("--log_batch", type=int, default=100)

    args = parser.parse_args()

    return args
