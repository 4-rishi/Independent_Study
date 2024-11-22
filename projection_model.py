import torch
import random
from torch.utils.data import Dataset
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
 
def generate_hyponyms(path):
    with open(path, "r", encoding = "utf-8") as f:
        lines = f.readlines()
        
    hyponyms = []
    for l in lines:
        ws, ty = l.strip().split("\t")
        w1 = "_".join(ws.split(" ")).lower()
        hyponyms.append(w1)    
    return hyponyms

#Function to generate hypernyms corresponding to the hyponyms(line by line) 
def generate_hypernyms(path):
    with open(path, "r", encoding = "utf-8") as f:
        lines = f.readlines()
    
    hypernyms = []
    for l in lines:
        ws = l.strip().split("\t")
        m= []
        for w in ws:
            w1 = "_".join(w.split(" ")).lower()
            m.append(w1)
        hypernyms.append(m)
                
    return hypernyms

def gen_all_hypernyms(hypernyms):
        all_ = []
        for hypernym in hypernyms:
            all_.extend(hypernym)
        temp_set = set(all_)
        return list(temp_set)

class HypernymDataset(Dataset):
    
    def __init__(self, vocab_file, data_file):
        # Load vocabulary
        with open(vocab_file, 'r') as f:
            self.vocab = {word.strip(): i for i, word in enumerate(f)}

        # Load data (assuming each line is "hyponym hypernym")
        with open(data_file, 'r') as f:
            self.data = [line.strip().split() for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        hyponym, hypernym = self.data[index]
        hyponym_idx = self.vocab[hyponym]
        hypernym_idx = self.vocab[hypernym]

        # Sample negative examples (1:nb_neg_samples ratio)
        negative_hypernym_idxs = []
        nb_neg_samples = 10
        while len(negative_hypernym_idxs) < nb_neg_samples:
            cand_id = random.randint(0, len(self.vocab) - 1)
            if cand_id != hypernym_idx:
                negative_hypernym_idxs.append(cand_id)

        # Create positive example
        positive_example = (hyponym_idx, hypernym_idx, 1)

        # Create negative examples
        negative_examples = [
            (hyponym_idx, neg_hypernym_idx, 0)
            for neg_hypernym_idx in negative_hypernym_idxs
        ]

        # Combine positive and negative examples
        all_examples = [positive_example] + negative_examples

        return all_examples

    def collate_fn(self, batch):
        # Flatten the batch
        flattened_batch = [item for sublist in batch for item in sublist]
        hyponym_idx, hypernym_idx, labels = zip(*flattened_batch)

        return (
            torch.tensor(hyponym_idx),
            torch.tensor(hypernym_idx),
            torch.tensor(labels),
        )

def get_vocab_size(vocab_file):
    with open(vocab_file, 'r') as f:
        return sum(1 for _ in f)   

class AllPairsHypernymDataset(Dataset):
    def __init__(self, vocab_file, hyponym_file, hypernym_file):
        # Load vocabulary
        with open(vocab_file, 'r') as f:
            self.vocab = {word.strip(): i for i, word in enumerate(f)}

        # Load data (assuming each line is "hyponym hypernym")
        with open(hyponym_file, 'r') as f:
            self.hyponyms = [line.strip() for line in f]

        with open(hypernym_file, 'r') as f:
            self.hypernyms = [line.strip() for line in f]

        # Create all pairs of hyponym-hypernym
        self.all_pairs = []
        for hyponym in self.hyponyms:
            hyponym_idx = self.vocab[hyponym]
            for hypernym in self.hypernyms:
                hypernym_idx = self.vocab[hypernym]
                self.all_pairs.append((hyponym_idx, hypernym_idx))

    def __len__(self):
        return len(self.all_pairs)  # Return the total number of pairs

    def __getitem__(self, index):
        return self.all_pairs[index]  # Return a tuple of (hyponym_idx, hypernym_idx)

    def collate_fn(self, batch):
        return (
            torch.tensor([pair[0] for pair in batch]),
            torch.tensor([pair[1] for pair in batch])
        )


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
    
    vocab_file = domain+"_vocab.txt"
    


    
    vocab = hyponyms_train
    vocab += hyponyms_trial
    vocab += hyponyms_test
    vocab += gen_all_hypernyms(hypernyms_train)
    vocab += gen_all_hypernyms(hypernyms_trial)
    vocab += gen_all_hypernyms(hypernyms_test)
    vocab = list(set(vocab))
    
    all_hypernyms_list = gen_all_hypernyms(hypernyms_train)
    all_hypernyms_list += gen_all_hypernyms(hypernyms_trial)
    all_hypernyms_list += gen_all_hypernyms(hypernyms_test)
    
    
    file = open(vocab_file, "w")
    for word in vocab:
        file.write(word + "\n")
    file.close()
    
    with open(domain+"_train_hypernym_data.txt", "w") as outfile:
        for hyponym, hypernym_list in zip(hyponyms_train, hypernyms_train):
            for hypernym in hypernym_list:
                outfile.write(f"{hyponym} {hypernym}\n")
    outfile.close()
    
    with open(domain+"_all_hypernyms_list.txt", "w") as outfile:
        for hypernym in all_hypernyms_list:
            outfile.write(f"{hypernym}\n")
    outfile.close()
    
    
    #word_embeddings = np.load('/kaggle/input/word-embeddings/englishembeddings.npy')
    #word_embeddings = np.load('/kaggle/input/italian-embeddings/italianembeddings.npy')
    #word_embeddings = np.load('/kaggle/input/spanish-embeddings/spanishembeddings.npy')
    word_embeddings = np.load('SemEval2018-Task9/New folder/medicalembeddings.npy')
    print(word_embeddings.shape)

    embedding_dim = 300
    vocab_size = len(vocab)


    embedding_layer = nn.Embedding.from_pretrained(torch.FloatTensor(word_embeddings), freeze=True)

    class HypernymModel(nn.Module):
        def __init__(self, embedding_dim, num_projections, vocab_size, dropout_rate):
            super(HypernymModel, self).__init__()
            #self.embeddings = nn.Embedding(vocab_size, embedding_dim)
            self.embeddings = embedding_layer
            # Initialize projection matrices with noise added to identity matrix
            self.projections = nn.Parameter(torch.eye(embedding_dim).repeat(num_projections, 1, 1) + torch.randn(num_projections, embedding_dim, embedding_dim) * 0.1) 
            self.linear = nn.Linear(num_projections, 1)
            self.sigmoid = nn.Sigmoid()
            self.dropout = nn.Dropout(dropout_rate)

        def forward(self, query, candidate):
            # Normalize embeddings
            query_emb = self.embeddings(query)

            query_emb = query_emb / torch.norm(query_emb, dim=1, keepdim=True)

            candidate_emb = self.embeddings(candidate)

            candidate_emb = candidate_emb / torch.norm(candidate_emb, dim=1, keepdim=True)
            
            all_projections = []

            # Compute projections for each projection matrix
            for proj_idx in range(self.projections.shape[0]):
                projection_matrix = self.projections[proj_idx]  # Get one projection matrix
                projection = torch.matmul(projection_matrix, query_emb.transpose(0, 1))  # (300, 1)
                all_projections.append(projection.transpose(0, 1))  # Transpose to (1, 300)

            concatenated_projections = torch.stack(all_projections, dim=1)  # (1, 24, 300)

            # Squeeze the second dimension to get the final output
            projections = concatenated_projections.squeeze(1)  # (24, 300)
            
            projections = self.dropout(projections)
            #print(projections.shape)


            # Calculate similarity scores
            scores = torch.matmul(projections, candidate_emb.unsqueeze(2)).squeeze(2)
            #print(scores.shape)

            # Apply dropout to embeddings and scores
            query_emb = self.dropout(query_emb)
            candidate_emb = self.dropout(candidate_emb)
            scores = self.dropout(scores)

            # Final output
            output = self.sigmoid(self.linear(scores))
            #print(output.shape)
            return output

    root1 = './'
    embedding_dim = 300
    num_projections = 24
    vocab_size = len(vocab)
    dropout_rate = 0.5
    learning_rate = 2e-4
    batch_size = 32
    num_epochs = 1
    
    dataset = HypernymDataset(root1+domain+"_vocab.txt", root1+domain+"_train_hypernym_data.txt")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate_fn)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    
    model = HypernymModel(embedding_dim, num_projections, vocab_size, dropout_rate)
    model.to(device)  # Move the model to GPU

    print(model)
    # Create optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCELoss()

    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = len(dataloader)

        for i, batch in enumerate(dataloader):
            # Get data
            query, candidate, labels = batch

            # Move data to GPU
            query, candidate, labels = query.to(device), candidate.to(device), labels.to(device)

            # Forward pass
            output = model(query, candidate)

            # Calculate loss
            loss = loss_fn(output.squeeze(1), labels.float())
            total_loss += loss.item()

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Calculate average loss for this epoch
        avg_loss = total_loss / num_batches
        print(f"Epoch [{epoch+1}/{num_epochs}] - Avg Loss: {avg_loss:.4f}")

    # Training completed
    print("Training finished!")

    # Save the entire model (including parameters and architecture)
    torch.save(model.state_dict(), domain+'hypernym_model.pth')
    
    loaded_model = HypernymModel(embedding_dim, num_projections, vocab_size, dropout_rate)
    #loaded_model.load_state_dict(torch.load('/kaggle/input/hypernym-model/'+domain+'hypernym_model.pth',map_location=device))
    loaded_model.load_state_dict(torch.load(root1+domain+'hypernym_model.pth',map_location=device))
    
    with open(domain+"_test_"+"hypernym_data.txt", "w") as outfile:
        for hyponym in hyponyms_test:
            outfile.write(f"{hyponym}\n")
    outfile.close()
    
    test_dataset = AllPairsHypernymDataset(root1+domain+"_vocab.txt", root1+domain+"_test_"+"hypernym_data.txt",root1+domain+"_all_hypernyms_list.txt")
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=test_dataset.collate_fn)


    loaded_model = loaded_model.to(device)  # Move model to GPU
    loaded_model.embeddings = loaded_model.embeddings.to(device)
    loaded_model.eval()

    hyponym_to_hypernym_output = {}


    for hyponyms,hypernyms in test_dataloader:
        hyponyms,hypernyms = hyponyms.to(device),hypernyms.to(device)
        #print(device)
        output = loaded_model(hyponyms, hypernyms)
        
        for i in range(len(hyponyms)):
            hyponym = hyponyms[i].item()  # Convert tensor to integer
            hypernym = hypernyms[i].item()  # Convert tensor to integer
            output_value = output[i].item()  # Convert tensor to float (or whatever type your output is)

            # If the hyponym is not already in the dictionary, add it
            if hyponym not in hyponym_to_hypernym_output:
                hyponym_to_hypernym_output[hyponym] = []

            # Append the (hypernym, output_value) tuple to the list
            hyponym_to_hypernym_output[hyponym].append((hypernym, output_value))
        break
    
    print(len(hyponym_to_hypernym_output)) 
    
    for hyponym, pairs in hyponym_to_hypernym_output.items():
        sorted_pairs = sorted(pairs, key=lambda x: x[1],reverse=True)  # Sort by output value
        hyponym_to_hypernym_output[hyponym] = sorted_pairs[:100]
        
    def retrieve_hypernym_words(hyponym_to_hypernym_output, idx2word):
        answer = []
        
        for hyponym_id, pairs in hyponym_to_hypernym_output.items():
            #sample_list.append(idx2word.get(hyponym_id))
            hypernyms_list = []
            for hypernym_id, score in pairs:
                # Get hypernym word from idx2word
                hypernym_word = idx2word.get(hypernym_id)
                #print(hypernym_word,hypernym_id)
                # Replace underscores with spaces
                hypernym_word = hypernym_word.replace("_", " ")
                #print(hypernym_word)
                
                
                hypernyms_list.append(hypernym_word)
                #print(hypernyms_list)
            answer.append(hypernyms_list)
            #print(answer)
        return answer

    def store_pairs_in_text_file(answer, output_file):
        with open(output_file, 'w') as f:
            for list_ in answer:
                for hypernym in list_:
                    f.write(hypernym + "\t")
                f.write("\n")



    with open(vocab_file, 'r') as f:
        idx2word = {i:word.strip() for i,word in enumerate(f)}
        
    output_file = domain+"_predicted.txt"
    answer = retrieve_hypernym_words(hyponym_to_hypernym_output, idx2word)
    store_pairs_in_text_file(answer, output_file)
    print(f"Formatted pairs saved in {output_file}")

    
    
    
    


    
    

    



    
    