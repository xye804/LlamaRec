import torch.nn as nn
import torch
import torch.nn.functional as F
from parameters import parse_args

args = parse_args()

def encode_text(llama, tokenizer, text_list):
    inputs = tokenizer(text_list, return_tensors="pt", padding=True, truncation=True).to("cuda")
    with torch.no_grad():
        outputs = llama(**inputs, output_hidden_states=True)
    return outputs.hidden_states[-1]


class AttentionPooling(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(AttentionPooling, self).__init__()
        self.att_fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.att_fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x, attn_mask=None):
        bz = x.shape[0]
        e = self.att_fc1(x)
        e = nn.Tanh()(e)
        alpha = self.att_fc2(e)
        alpha = torch.exp(alpha)
        if attn_mask is not None:
            alpha = alpha * attn_mask.unsqueeze(2)
        alpha = alpha / (torch.sum(alpha, dim=1, keepdim=True) + 1e-8)
        x = torch.bmm(x.permute(0, 2, 1), alpha)
        x = torch.reshape(x, (bz, -1))
        return x


class NewsEncoder(nn.Module):
    def __init__(self, llama, tokenizer, embedding_dim, hidden_dim):
        super(NewsEncoder, self).__init__()
        self.llama = llama
        self.tokenizer = tokenizer
        self.att_pool1 = AttentionPooling(embedding_dim, hidden_dim)
        self.att_pool2 = AttentionPooling(embedding_dim * 2, hidden_dim)
        self.fc = nn.Linear(embedding_dim * 2, hidden_dim)

    def forward(self, candidate_texts):
        candidate_embeddings = []
        for candidate_list in candidate_texts:
            last_state = encode_text(self.llama, self.tokenizer, candidate_list)
            mask = torch.ones(last_state.shape[0], last_state.shape[1]).bool().to("cuda")
            news_level_repr = self.att_pool1(last_state.float(), mask)
            extended_news_level_repr = news_level_repr.unsqueeze(1).expand(-1, last_state.shape[1], -1)
            word_level_repr = last_state
            candidate_embedding = torch.cat((extended_news_level_repr, word_level_repr), dim=-1)
            mask = torch.ones(candidate_embedding.shape[0], candidate_embedding.shape[1]).bool().to("cuda")
            candidate_embedding = self.att_pool2(candidate_embedding.float(), mask)
            candidate_embedding = self.fc(candidate_embedding)
            candidate_embeddings.append(candidate_embedding)
        return candidate_embeddings


class Fastformer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(Fastformer, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim % num_head should be 0"

        self.query_proj = nn.Linear(embed_dim, embed_dim) 
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)

        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()

        queries = self.query_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        keys = self.key_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        values = self.value_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        qk_sum = torch.sum(queries + keys, dim=1)

        attn_output = torch.softmax(qk_sum, dim=-1)
        attn_output = attn_output.unsqueeze(1) * values
        attn_output = torch.sum(attn_output, dim=1)
        attn_output = attn_output.view(batch_size, -1)

        output = self.feedforward(attn_output)
        return output


class UserEncoder(nn.Module):
    def __init__(self, llama, tokenizer, embedding_dim, hidden_dim):
        super(UserEncoder, self).__init__()
        self.llama = llama
        self.tokenizer = tokenizer
        self.fast_att = Fastformer(embedding_dim * 2, 8)
        self.att_pool1 = AttentionPooling(embedding_dim, hidden_dim)
        self.att_pool2 = AttentionPooling(embedding_dim * 2, hidden_dim)
        self.fc = nn.Linear(embedding_dim * 2, hidden_dim)

    def forward(self, user_history_texts):
        user_interests = []
        for user_history_list in user_history_texts:
            last_state = encode_text(self.llama, self.tokenizer, user_history_list)
            mask = torch.ones(last_state.shape[0], last_state.shape[1]).bool().to("cuda")
            news_level_repr = self.att_pool1(last_state.float(), mask)
            extended_news_level_repr = news_level_repr.unsqueeze(1).expand(-1, last_state.shape[1], -1)
            word_level_repr = last_state
            his_embedding = torch.cat((extended_news_level_repr, word_level_repr), dim=-1)
            mask = torch.ones(his_embedding.shape[0], his_embedding.shape[1]).bool().to("cuda")
            his_embedding = self.att_pool2(his_embedding.float(), mask).unsqueeze(0)
            user_interest = self.fast_att(his_embedding).squeeze(0)
            user_interest = self.fc(user_interest)
            user_interests.append(user_interest)
        return user_interests


class CTRModel(nn.Module):
    def __init__(self, hidden_dim):
        super(CTRModel, self).__init__()
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)

    def forward(self, user_interest, candidate_embeddings):
        x = torch.cat([user_interest, candidate_embeddings], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class LlamaRec(nn.Module):
    def __init__(self, llama, tokenizer, embedding_dim, hidden_dim):
        super(LlamaRec, self).__init__()
        self.llama = llama
        self.user_encoder = UserEncoder(llama, tokenizer, embedding_dim, hidden_dim)
        self.news_encoder = NewsEncoder(llama, tokenizer, embedding_dim, hidden_dim)
        self.ctr_model = CTRModel(hidden_dim)

    def forward(self, user_history_texts, candidate_texts):
        user_embeddings = self.user_encoder(user_history_texts)
        candidate_embeddings = self.news_encoder(candidate_texts)
        probabilitiess = []
        for i in range(len(candidate_embeddings)):
            candidate_embedding_tuple = torch.unbind(candidate_embeddings[i])
            probabilities = [self.ctr_model(user_embeddings[i], candidate_embedding) for candidate_embedding in
                             candidate_embedding_tuple]
            probabilitiess.append(probabilities)
        return probabilitiess
