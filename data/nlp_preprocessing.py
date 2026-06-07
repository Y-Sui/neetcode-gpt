import torch
import torch.nn as nn
from torchtyping import TensorType

# torch.tensor(python_list) returns a Python list as a tensor
class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        words = set()
        combined = positive + negative
        for sentence in combined:
            for word in sentence.split():
                words.add(word)

        # vocab mapping from word to int
        sorted_list = sorted(list(words))
        word_to_int = {}
        for i, c in enumerate(sorted_list):
            word_to_int[c] = i + 1

        # convert the sentence to int list
        def encode(sentence):
            integers = []
            for word in sentence.split():
                integers.append(word_to_int[word])
            return integers

        var_len_tensors = []
        for sentence in combined:
            var_len_tensors.append(torch.tensor(encode(sentence)))


        # 核心函数：pad_sequence
        # 这个函数会自动找到最长的 Tensor，然后把其他短的 Tensor 用 0 补齐
        # batch_first=True 意思是输出形状为 (句子数量, 最大长度)，符合人类直觉
        return nn.utils.rnn.pad_sequence(var_len_tensors, batch_first=True)

