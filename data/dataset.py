import torch
from typing import List, Tuple

class Solution:
    def batch_loader(self, raw_dataset: str, context_length: int, batch_size: int) -> Tuple[List[List[str]]]:
        # You must start by generating batch_size different random indices in the appropriate range
        # using a single call to torch.randint()
        torch.manual_seed(0)
        tokenized = raw_dataset.split()
        # index 最大应该小于len - context length以免越界
        indices = torch.randint(
            low=0, 
            high=len(tokenized)-context_length, 
            size=(batch_size,)
        ).tolist()
        X, Y = [], []
        for idx in indices:
            X.append(tokenized[idx:idx+context_length])
            Y.append(tokenized[idx+1:idx+1+context_length])
        return X, Y
