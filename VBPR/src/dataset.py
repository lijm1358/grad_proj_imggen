import numpy as np
import random
from tqdm import tqdm 
from torch.utils.data import Dataset

class HMTrainDataset(Dataset):
    def __init__(self, df, item_df, items_by_prod_type, pos_items_each_user) -> None:
        super().__init__()
        self.df = df
        self.item_df = item_df
        self.items_by_prod_type = items_by_prod_type
        self.pos_items_each_user = pos_items_each_user
        self.df['neg'] = np.zeros(len(self.df), dtype=int)
        self._make_triples_data()
    
    def __getitem__(self, index):
        user = self.df.customer_id[index]
        pos = self.df.article_id[index]
        neg = self.df.neg[index]
        return user, pos, neg
            
    def _neg_sampling(self, pos_list, prod_type_no):
        # 같은 prod_type_no 내에서 neg sampling
        neg = random.choice(self.items_by_prod_type[prod_type_no]) 
        while neg in pos_list:
            neg = random.choice(self.items_by_prod_type[prod_type_no])
        return neg

    def _make_triples_data(self):
        for user_id, rows in tqdm(self.df.groupby("customer_id")):
            # pos_list = {k:1 for k in pos_items_each_user[user_id]}
            pos_list = self.pos_items_each_user[user_id]
            for idx, row in rows.iterrows():
                item_id = row.article_id
                prod_type_no = self.item_df[self.item_df["article_id"] == item_id].product_type_no.item()
                self.df.at[idx, 'neg'] = self._neg_sampling(pos_list, prod_type_no)
    
    def __len__(self):
        return len(self.df)
    

class HMTestDataset(Dataset):
    def __init__(self, df) -> None:
        super().__init__()
        self.df = df
        
    def __getitem__(self, index):
        user = self.df.customer_id[index]
        pos = self.df.article_id[index]
        return user, pos

    def __len__(self):
        return len(self.df)