from torch.utils.data import Dataset
import pandas as pd


class LlamaRecDataset(Dataset):
    def __init__(self, news_df, behaviors_df, args, test=False):
        self.behaviors_df = behaviors_df
        self.news_df = news_df
        self.test = test
        self.dataset = None
        self.np_ratio = args.np_ratio
        self.process()

    def process(self):

        def format_news(news_id):
            news_list = []
            for id in news_id:
                if id in news_df.index:
                    news = news_df.loc[id]
                    news_list.append(news['Title'])
            return news_list

        history = []
        candidate = []
        labels = []
        news_df = self.news_df.set_index('NewsId')
        for _, row in self.behaviors_df.iterrows():
            history.append(format_news(row['History']))
            if self.test:
                clicked_news_ids = [news_id.split('-')[0] for news_id in row['Impressions'] if news_id.endswith('-1')]
                non_clicked_news_ids = [news_id.split('-')[0] for news_id in row['Impressions'] if
                                       news_id.endswith('-0')]
            else:
                clicked_news_ids = [news_id.split('-')[0] for news_id in row['Impressions'] if news_id.endswith('-1')]
                non_clicked_news_ids = [news_id.split('-')[0] for news_id in row['Impressions'] if
                                       news_id.endswith('-0')][:self.np_ratio * len(clicked_news_ids)]
            positive = format_news(clicked_news_ids)
            negative = format_news(non_clicked_news_ids)
            label = len(positive) * [1] + len(negative) * [0]
            candidate.append(positive + negative)
            labels.append(label)
        self.dataset = pd.DataFrame({'History': history,
                                     'Candidate': candidate,
                                     'Label': labels})

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset.iloc[idx]
        return row['History'], row['Candidate'], row["Label"]
