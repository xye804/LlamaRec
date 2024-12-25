from sklearn.model_selection import train_test_split
import pandas as pd


class DataPreprocess:
    def __init__(self, mode):
        self.mode = mode
        if mode == "train":
            self.news_path = "data/train_news.tsv"
            self.behaviors_path = "data/train_behaviors.tsv"
        else:
            self.news_path = "data/dev_news.tsv"
            self.behaviors_path = "data/dev_behaviors.tsv"

    def process(self):
        column_news = ['NewsId', 'Category', 'SubCat', 'Title', 'Abstract', 'url', 'TitleEnt', 'AbstractEnt']
        column_behaviors = ['ImpressionId', 'UserId', 'Time', 'History', 'Impressions']
        news_df = pd.read_csv(self.news_path, sep='\t', header=None, names=column_news)
        behaviors_df = pd.read_csv(self.behaviors_path, sep='\t', header=None, names=column_behaviors)

        news_df = news_df.drop(columns=['Abstract', 'url', 'Category', 'SubCat', 'TitleEnt', 'AbstractEnt']).dropna()
        behaviors_df = behaviors_df.drop(columns=['ImpressionId', 'Time', 'UserId']).dropna()

        behaviors_df['Impressions'] = behaviors_df['Impressions'].apply(lambda x: x.split(" ")).dropna()
        behaviors_df['History'] = behaviors_df['History'].apply(lambda x: x.split(" ")).dropna()

        behaviors_df = behaviors_df[behaviors_df['History'].apply(lambda x: 5 <= len(x) <= 40)]

        if self.mode != "train":
            val_behaviors_df, test_behaviors_df = train_test_split(behaviors_df, test_size=0.5, random_state=42)
            return news_df, val_behaviors_df, test_behaviors_df
        else:
            return news_df, behaviors_df
