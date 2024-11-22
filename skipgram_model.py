import torch
import torch.nn as nn
import torch.optim as optim

from create_dataset import SkipGramDataset,read_vocab_file
from vocab import generate_hypernyms,generate_hyponyms

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

class SkipGramModel(nn.Module):
    def __init__(self, emb_size, emb_dimension):
        super().__init__()
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension
        self.u_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)
        self.init_weights()

    def init_weights(self):
        initrange = 1.0 / self.emb_dimension
        init.uniform_(self.u_embeddings.weight.data, -initrange, initrange)

    def forward(self, pos_u, pos_v, neg_v):
        emb_u = self.u_embeddings(pos_u)
        emb_v = self.u_embeddings(pos_v)
        emb_neg_v = self.u_embeddings(neg_v)

        score = torch.sum(torch.mul(emb_u, emb_v), dim=1)
        score = F.logsigmoid(score)

        neg_score = torch.bmm(emb_neg_v, emb_u.unsqueeze(2)).squeeze()
        neg_score = F.logsigmoid(-neg_score)

        return - (torch.mean(score) + torch.mean(neg_score)) 


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
   
    vocab_size = len(word_list)
    dataset = SkipGramDataset(hyponyms_train,hypernyms_train,word_list)
    dataloader = DataLoader(dataset, batch_size=32,
                                     shuffle=True, num_workers=10, collate_fn=dataset.collate)
    
    embedding_dim = 300  # Set your desired embedding dimension
    model = SkipGramModel(vocab_size, embedding_dim)

    
    
    epochs = 200
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for epoch in tqdm(range(epochs)):
        #optimizer = optim.SGD(model.parameters(), lr=0.01)
        optimizer = optim.SparseAdam(model.parameters(), lr=0.001)
        running_loss = 0.0
        for i, sample_batched in enumerate(dataloader):
            if len(sample_batched[0]) > 1:
                pos_u = sample_batched[0].to(device)
                pos_v = sample_batched[1].to(device)
                neg_v = sample_batched[2].to(device)
                optimizer.zero_grad()
                loss = model.forward(pos_u, pos_v, neg_v)
                loss.backward()
                optimizer.step()
                running_loss = running_loss * 0.9 + loss.item() * 0.1
        print(running_loss)
        
    embedding = model.u_embeddings.weight.cpu().data.numpy()
    
    np.save(root+domain+'embeddings',embedding)
    print(len(embedding[0]),len(embedding))

    
    

