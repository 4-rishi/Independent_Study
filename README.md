# Independent_Study

### 1.Created the comprehensive vocab for each task data
### 2.Created the dataset, dataloader from the data
### 3. Trained the skipgram model with negative sampling for getting embeddings of the words
### 4. Did a simple cosine similarity search for 15 nearest words
### 5. Loaded word embeddings into projection model, and trained the projection model and generated the 100 nearest words based on sigmoid score
### 6. And evaluated the nearest words against the gold standard words using the task9-scorer.py to see the performance.

### To create vocabulary run `python vocab.py`
### To preprocess and create a datset run `python create_dataset.py`
### To run skipgram model  `python skipgram_model.py`
### To generate hypernymns from cosine similarity `python cosine_similarity.py`
### To generate hypernymns from projection model `python projection_model.py`

## Requirements
### Numpy,pytorch,sklearn
