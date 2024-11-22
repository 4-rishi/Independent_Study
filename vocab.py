def generate_hyponyms(path):
    with open(path, "r", encoding = "utf-8") as f:
        lines = f.readlines()
        
    hyponyms = []
    for l in lines:
        #print(l.strip().split("\t"))
        ws, ty = l.strip().split('\t')
        w1 = "_".join(ws.split(" ")).lower()
        hyponyms.append(w1)    
    return hyponyms


# def generate_hyponyms(path):
#     with open(path, "r", encoding="utf-8") as f:
#         lines = f.readlines()
        
#     hyponyms = []
#     for l in lines:
#         parts = l.strip().split("\t")
#         w1 = "_".join(parts[0].split(" ")).lower()
#         if len(parts) > 2:
#             w2 = "_".join(parts[1:-1]).lower()
#             hyponyms.append((w1, w2))
#         else:
#             w2 = parts[1].lower()
#             hyponyms.append((w1, w2))
#     return hyponyms

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
    
    vocab = hyponyms_train
    vocab += hyponyms_trial
    vocab += hyponyms_test
    vocab += gen_all_hypernyms(hypernyms_train)
    vocab += gen_all_hypernyms(hypernyms_trial)
    vocab += gen_all_hypernyms(hypernyms_test)
    vocab = list(set(vocab))
    
    
    file = open(root+vocab_file, "w")
    for word in vocab:
        #print(type(word))
        #break
        file.write(word[0] + "\n")
        file.write(word + "\n")
    file.close()
    
    

    

