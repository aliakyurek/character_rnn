"""
This is the pytorch dataset and Pytorch Lightning data module classes for character generation task.
Basically, we have a text file and we want to generate similar text using LSTM based character generator model.
"""
import random
import re
import string
import torch
from torch.utils import data
import pytorch_lightning as pl


class CharacterDataSet(data.IterableDataset):
    """
    This class is a Pytorch IterableDataset class to handle character generation task.
    It reads the text file, tokenizes the characters, and yields the input and target tensors.
    IterableDataset fits best to the character generation task because we can generate infinite data from a corpus
    and we don't need to implement __len__ and __getitem__ methods.
    """

    def __init__(self, file_path, portion_size, iters_per_epoch, **kwargs):
        """
        Initializes the dataset.
        :param file_path: Path to the text file.
        :param portion_size: Size of the portion to be copied from the text file content.
        :param iters_per_epoch: Number of iterations to yield in each epoch.
        """
        self.portion_size = portion_size
        self.iters_per_epoch = iters_per_epoch
        with open(file_path, 'r', encoding='utf-8') as f:
            self.file_content = f.read()

        # update the class member vocab with the unique characters in the text file.
        CharacterDataSet.vocab = sorted(set(self.file_content))

    def __iter__(self):
        # in each epoch, we get random portions from text. So to have a limit, we use iters_per_epoch to stop yielding.
        for i in range(self.iters_per_epoch):
            portion = self.random_portion()  # "r second vaccine dose..."
            tokenized = self.char_to_tensor(portion)  # tokenize and tensorize
            # one character shift to create input, target pairs. "programmin" -> "rogramming"
            inputs = tokenized[:-1]
            targets = tokenized[1:]
            yield inputs, targets

    def random_portion(self):
        """
        This method returns a random portion of the text file content.
        """
        start_index = random.randint(0, len(self.file_content) - self.portion_size)

        # if the start_index is not at the beginning of a word, find the beginning of the word.
        # if it already points to a space or dot, no need to find the beginning of the word but just increment the index.
        while self.file_content[start_index] not in (' ', '.') and start_index >= 0:
            start_index -= 1
        
        start_index += 1

        end_index = start_index + self.portion_size + 1
        return self.file_content[start_index:end_index]

    @staticmethod
    def char_to_tensor(text):
        """
        This method converts the text to a tensor of character indexes.
        :param text: The text to be converted.
        """
        lst = [CharacterDataSet.vocab.index(c) for c in text]
        tensor = torch.tensor(lst, dtype=torch.long)
        return tensor

class NoBatchingDataModule(pl.LightningDataModule):
    """
    This is the PL DataModule class to handle the character generation task.
    PL DataModule class is used to encapsulate all the data related operations like data loading, splitting, and preprocessing
    It provides data loaders for training, validation, and test data and make them reusable in the applications
    """
    def __init__(self, dataset):
        super().__init__()
        self.train_dataset = dataset
        
    def setup(self, stage: str) -> None:
        # create the dataset instance
        pass

    def train_dataloader(self):
        # when we set batch_size to None, the DataLoader will rely on the __iter__ method of the dataset, just as we want.
        return data.DataLoader(self.train_dataset, batch_size=None)
    
class BatchingDataModule(pl.LightningDataModule):
    """
    This is the PL DataModule class to handle the character generation task.
    PL DataModule class is used to encapsulate all the data related operations like data loading, splitting, and preprocessing
    It provides data loaders for training, validation, and test data and make them reusable in the applications
    """
    def __init__(self, dataset, batch_size):
        super().__init__()
        self.train_dataset = dataset
        self.batch_size = batch_size
        
    def setup(self, stage: str) -> None:
        # create the dataset instance
        pass

    def train_dataloader(self):
        return data.DataLoader(self.train_dataset, batch_size=self.batch_size)
