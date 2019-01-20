"""
Dataset class adapted from https://github.com/nmhkahn/MemN2N-pytorch/blob/master/memn2n/dataset.py
"""
import torch
import torch.utils.data as data
from data_utils import *#load_task, vectorize_data

class bAbIDataset(data.Dataset):
    def __init__(self, dataset_dir, task_id=1, memory_size=70, train=True):
        self.train = train
        self.task_id = task_id
        self.dataset_dir = dataset_dir

        train_data, test_data = load_task(self.dataset_dir, task_id)
        data = train_data + test_data


        if task_id == 'qa3':
            truncated_story_length = 130
        else:
            truncated_story_length = 70
        stories_train = truncate_stories(train_data, truncated_story_length)
        stories_test = truncate_stories(test_data, truncated_story_length)

        self.vocab, token_to_id = get_tokenizer(stories_train + stories_test)
        self.num_vocab = len(self.vocab)

        stories_token_train = tokenize_stories(stories_train, token_to_id)
        stories_token_test = tokenize_stories(stories_test, token_to_id)
        stories_token_all = stories_token_train + stories_token_test


        story_lengths = [len(sentence) for story, _, _ in stories_token_all for sentence in story]
        max_sentence_length = max(story_lengths)
        max_story_length = max([len(story) for story, _, _ in stories_token_all])
        max_query_length = max([len(query) for _, query, _ in stories_token_all])
        self.sentence_size = max_sentence_length
        self.query_size = max_query_length
        if train:
            story, query, answer = pad_stories(stories_token_train, \
                max_sentence_length, max_story_length, max_query_length)
        else:

            story, query, answer = pad_stories(stories_token_test, \
                max_sentence_length, max_story_length, max_query_length)
        self.data_story = torch.LongTensor(story)
        self.data_query = torch.LongTensor(query)
        self.data_answer = torch.LongTensor(answer)

    def __getitem__(self, idx):
        return self.data_story[idx], self.data_query[idx], self.data_answer[idx]

    def __len__(self):
        return len(self.data_story)
