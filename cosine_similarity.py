import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset
from torch.nn import init
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from create_dataset import SkipGramDataset,read_vocab_file
from vocab import generate_hypernyms,generate_hyponyms


from sklearn.metrics.pairwise import cosine_similarity




if __name__ == '__main__':
#     p = argparse.ArgumentParser()
#     p.add_argument('domain',type=str)   
#     domain = p.parse_args().domain
    domain = "english"
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
        #vocab_path = "vocabulary/2A.medical.vocabulary.txt"
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
    #dataset = SkipGramDataset(hyponyms_trial,hypernyms_trial,word_list)
    #dataset = SkipGramDataset(hyponyms_test,hypernyms_test,word_list)
    
    embeddings = np.load(root+domain+'embeddings.npy')
    vocab_size, embedding_dim = embeddings.shape  # Get dimensions from loaded data

    embedding_layer = nn.Embedding(vocab_size, embedding_dim)
    embedding_layer.weight = nn.Parameter(torch.tensor(embeddings))  # Load pre-trained weights

    # Optionally freeze the embeddings
    embedding_layer.weight.requires_grad = False  # Prevent updates during training
    
    print(embeddings[0].shape,dataset.idx2word[0],dataset.word2idx['racing_boat'])
    
    print(embeddings[dataset.word2idx[hyponyms_train[0]]].shape,hyponyms_train[0])

    # Assuming you have a list of embeddings called "embeddings" and a target embedding
    # called "target_embedding" (e.g., embeddings[dataset.word2idx[hyponyms_train[0]]])

    # Calculate cosine similarity between the target embedding and all other embeddings
    nearest_words_dict = {}
    print(len(hyponyms_train))
    #print(len(hyponyms_test))
    # Iterate through all hyponyms_train
    #for hyponym in hyponyms_test:
    for hyponym in hyponyms_train:
  
        
        target_embedding = embeddings[dataset.word2idx[hyponym]]
        
        # Calculate cosine similarity between the target embedding and all other embeddings
        similarities = cosine_similarity(target_embedding.reshape(1, -1), embeddings)
        
        # Sort the similarities in descending order
        sorted_indices = np.argsort(similarities[0])[::-1]
        
        # Get the indices of the nearest 15 embeddings (excluding the target itself)
        nearest_indices = sorted_indices[1:16]
        
        # Get the nearest words
        nearest_words = [dataset.idx2word[idx] for idx in nearest_indices]
        
        if len(nearest_words) == 0:
            print("g")
        # Store the nearest words in the dictionary
        nearest_words_dict[hyponym] = nearest_words


    print(len(nearest_words_dict))

    
    with open(root+domain+"_train"+"_nearest_words.txt", "w") as f:
        for hyponym, nearest_words in nearest_words_dict.items():
            # Replace underscores with spaces
            nearest_words = [word.replace("_", " ") for word in nearest_words]
            f.write("\t".join(nearest_words) + "\n")

    print("Nearest words saved to nearest_words.txt")
