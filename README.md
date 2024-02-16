# News-Topic-Classification-Using-Transformer

Approach: Fine Tune only the final layer/ Head of Transformer Model

This approach is often used when the task for which the transformer model is being fine-tuned is different from the task for which it was pre-trained. ie only update the final layer to make predictions for the new task

### About Dataset

The AG's news topic classification dataset is constructed by choosing 4 largest classes from the original corpus. 
Each class contains 30,000 training samples and 1,900 testing samples. The total number of training samples is 120,000 and testing 7,600.

The first column is Class Id, the second column is Title and the third column is Description. \
The class ids are numbered 1-4 where 1 represents World, 2 represents Sports, 3 represents Business and 4 represents Sci/Tech.

### Tokenizer Model - DistilBERT

DistilBERT is a small, fast, cheap and light Transformer model trained by distilling BERT base. It has 40% less parameters than bert-base-uncased, runs 60% faster while preserving over 95% of BERT’s performances.

Here, we are using model 'distilbert-base-uncased'. So we use tokenizer for the DISTILBERT model with the "base" architecture and the "cased" version.


### Steps followed in this project

1. Read dataset
2. Clean Text data
3. Prepare Tokenizer
4. Initialize Model
5. Train Model
6. Evaluate Model



