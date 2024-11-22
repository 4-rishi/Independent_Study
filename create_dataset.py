from torch.utils.data import Dataset

import numpy as np
import random
from vocab import generate_hypernyms,generate_hyponyms,gen_all_hypernyms
import torch

def read_vocab_file(file_path):
    try:
        with open(file_path, 'r') as file:
            # Read lines and remove leading/trailing whitespaces
            words = [line.strip() for line in file.readlines()]
            return words
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return []


class SkipGramDataset(Dataset):
    def __init__(self, hyponyms, hypernyms, vocab, num_neg_samples=6):
        self.hyponyms = hyponyms
        self.hypernyms = hypernyms
        self.total_hypernyms = gen_all_hypernyms(hypernyms)
        self.vocab = vocab
        self.num_neg_samples = num_neg_samples
        self.word2idx = {word:idx for idx, word in enumerate(self.vocab) }
        self.idx2word = {idx:word for idx, word in enumerate(self.vocab) }               
                 


    def __create_items__(self,hyponym,hypernyms, negative_samples, word_id_map, num_negs):
        # Initialize an empty list to store the items
        items = []
        
        # Set the initial values for start and end indices
        start, end = 0, num_negs
        
        # Iterate over each hypernym
        for ndx, hypernym in enumerate(hypernyms):
            # Extract the corresponding negative samples
            negatives = negative_samples[start:end]
            
            # Update the start and end indices for the next iteration
            start = end
            end += num_negs
            
            # Create an item tuple with hyponym, hypernym, and negatives
            item = (word_id_map[hyponym], word_id_map[hypernym], [word_id_map[neg] for neg in negatives])
            
            # Append the item to the items list
            items.append(item)
        
        # Return the list of items
        return items

    def __getitem__(self, idx):
        hyponym = self.hyponyms[idx]
        hypernyms = self.hypernyms[idx]
        negative_hypernyms = []
        
        
        no_of_neg_samples = 0
        total_no_of_neg_samples =  len(hypernyms)*self.num_neg_samples
        
        ret = []
        
        while no_of_neg_samples < total_no_of_neg_samples:
            neg_idx = self.total_hypernyms[np.random.randint(0, len(self.total_hypernyms))]
            if neg_idx not in negative_hypernyms and neg_idx not in hypernyms:
                negative_hypernyms.append(neg_idx)
                no_of_neg_samples +=1
                
        return self.__create_items__(hyponym,hypernyms, negative_hypernyms, self.word2idx, self.num_neg_samples)  
          
    @staticmethod
    def collate(batches):
        u = [u for b in batches for u, _, _ in b if len(b) > 0]
        pos = [pos_v for b in batches for _, pos_v, _ in b if len(b) > 0]
        neg = [neg_v for b in batches for _, _, neg_v in b if len(b) > 0]
        return torch.LongTensor(u), torch.LongTensor(pos), torch.LongTensor(neg)        
    
    def __len__(self):
        return len(self.hyponyms)
    
if __name__ == '__main__':
#     p = argparse.ArgumentParser()
#     p.add_argument('domain',type=str)   
#     domain = p.parse_args().domain
    domain = "medical"
    root = "SemEval2018-Task9/"
    
    if(domain == "english"):
        vocab_path = "vocabulary/1A.english.vocabulary.txt"
        data_path_train = "training/data/1A.english.training.data.txt"
        data_path_trial = "trial/data/1A.english.trial.data.txt"
        data_path_test = "test/data/1A.english.test.data.txt"
        gold_path_train = "training/gold/1A.english.training.gold.txt"
        gold_path_trial = "trial/gold/1A.english.trial.gold.txt"
        gold_path_test =  "test/gold/1A.english.test.gold.txt"
    elif(domain == "italian"):
        vocab_path = "vocabulary/1B.italian.vocabulary.txt"
        data_path_train = "training/data/1B.italian.training.data.txt"
        data_path_trial = "trial/data/1B.italian.trial.data.txt"
        data_path_test = "test/data/1B.italian.test.data.txt"
        gold_path_train = "training/gold/1B.italian.training.gold.txt"
        gold_path_trial = "trial/gold/1B.italian.trial.gold.txt"
        gold_path_test =  "test/gold/1B.italian.test.gold.txt"
    elif(domain == "spanish"):
        vocab_path = "vocabulary/1C.spanish.vocabulary.txt"
        data_path_train = "training/data/1C.spanish.training.data.txt"
        data_path_trial = "trial/data/1C.spanish.trial.data.txt"
        data_path_test = "test/data/1C.spanish.test.data.txt"
        gold_path_train = "training/gold/1C.spanish.training.gold.txt"
        gold_path_trial = "trial/gold/1C.spanish.trial.gold.txt"
        gold_path_test =  "test/gold/1C.spanish.test.gold.txt"
    elif(domain == "medical"):
        vocab_path = "vocabulary/2A.medical.vocabulary.txt"
        data_path_train = "training/data/2A.medical.training.data.txt"
        data_path_trial = "trial/data/2A.medical.trial.data.txt"
        data_path_test = "test/data/2A.medical.test.data.txt"
        gold_path_train = "training/gold/2A.medical.training.gold.txt"
        gold_path_trial = "trial/gold/2A.medical.trial.gold.txt"
        gold_path_test =  "test/gold/2A.medical.test.gold.txt"
    elif(domain == "music"):
        vocab_path = "vocabulary/2B.music.vocabulary.txt"
        data_path_train = "training/data/2B.music.training.data.txt"
        data_path_trial = "trial/data/2B.music.trial.data.txt"
        data_path_test = "test/data/2B.music.test.data.txt"
        gold_path_train = "training/gold/2B.music.training.gold.txt"
        gold_path_trial = "trial/gold/2B.music.trial.gold.txt"
        gold_path_test =  "test/gold/2B.music.test.gold.txt"
    else:
        print("Incorrect argument")
        exit()


    hyponyms_train= generate_hyponyms(root+data_path_train)
    hyponyms_trial = generate_hyponyms(root+data_path_trial)
    hyponyms_test = generate_hyponyms(root+data_path_test)
    
    hypernyms_train = generate_hypernyms(root+gold_path_train)
    hypernyms_trial= generate_hypernyms(root+gold_path_trial)
    hypernyms_test = generate_hypernyms(root+gold_path_test)
    
    vocab_file = domain+"vocab.txt"
    

    word_list = read_vocab_file(root+vocab_file)
    print("vocab_size",len(word_list))
    
    dataset = SkipGramDataset(hyponyms_train,hypernyms_train,word_list)
    print("dataset",len(dataset),dataset[0])
    
    
     