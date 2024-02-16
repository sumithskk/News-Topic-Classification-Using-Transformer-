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
3. Prepare the Tokenizer
4. Initialize the Model
5. Train the Model
6. Evaluate the Model

### Data Preprocessing

There is both training and testing samples avaialbe from the source(Kaggle), So there is no need to split the data to train-test samples. 

Inititally we have three columns present in the samples, which are Class Id, Title and Description. As a first preprocessing step, we have combined both Title and Description as new column, which named as 'text' and the 'Class Id' is renamed to 'label'.

When we checked the class distributions of labels, there is no much imbalances present. So we don't need to any class imbalance processing on label column, and we can proceed with accuray as evalution metric.

The class ids are numbered 0-3 where

        0 - World,
        1 - Sports,
        2 - Business
        3 - Sci/Tech.

For the training we choose only 36k samples of data due to the computational limitations.  
 
### Prepapre Tokenizer

Convert the data into a numerical representation suitable for input into the Transformer model. This typically involves tokenizing the text into subwords or words, mapping the tokens to integers, and encoding the input as a tensor.

DistilBERT is a small, fast, cheap and light Transformer model trained by distilling BERT base. It has 40% less parameters than bert-base-uncased, runs 60% faster while preserving over 95% of BERT’s performances.

Here, we are using model 'distilbert-base-uncased'. So we use tokenizer for the DISTILBERT model with the "base" architecture and the "cased" version.

The from_pretrained() method takes care of returning the correct tokenizer class instance based on the model_type defined, here we have 'distilbert-base-uncased'. 

For the tokenizer, “truncation” argument is set to “True”, which means that the tokenization function will truncate sequences that are longer than the maximum length specified by the model.

### Train the Model

Since we will be using DistilBERT as our base model, we begin by importing distilbert-base-uncased from the Hugging Face library.

The fine-tuing approach: 
We have to decide which layers of the pre-trained model to fine-tune, here keeping the all the weights of the pre-trained model frozen and optimizing only the weights of the head layers ie. pre_classifier and classifier layer.  To temporarily freeze the pre-trained weights, set layer.trainable = False for each of layers. 
The percentage of trainble weights to the total weghits is 0.8% .

Before you start training your model, create a map of the expected ids to their labels with id2label and label2id:


id2label = {0: "World", 1: "Sports",2:"Business",3:'Sci/Tech'}
label2id = {"World":0, "Sports":1,"Business":2,'Sci/Tech':3}

'AutoModelForSequenceClassification' is a class from the transformers library that implements a sequence classification model, a type of model that is used to predict the class of a sequence of inputs (e.g., a sentence). 

'TrainingArguments' is a class that defines the arguments used to configure a training run. It includes arguments such as the number of training steps, the learning rate, the batch size, and many others. When using the Trainer class, an instance of TrainingArguments is passed to the constructor to specify the configuration for a training run.



##### TrainingArguments:

TrainingArguments set the arguments for training model. The output_dir argument sets the directory where the model and training-related files will be saved. The evaluation_strategy argument sets how often evaluation should be done, and in this case, it's set to be done every epoch.

The save_strategy argument sets when the model should be saved, and it's set to be saved every epoch. The num_train_epochs argument sets the number of training epochs, and it's set to 5. 

The per_device_train_batch_size argument sets the batch size for training, and it's set to 16. The per_device_eval_batch_size argument sets the batch size for evaluation, and it's set to 64.

##### Training:

Defined a function  for evaluation metrics, compute_metrics,that takes a tuple of logits_and_labels as input and computes two evaluation metrics: accuracy. This function first unpacks the tuple into logits and labels. Then it calculates the predictions using np.argmax along the last axis of logits.

The Trainer class takes several arguments:

* training_args is an instance of the TrainingArguments class that contains the arguments for training the model.
* train_dataset is the training dataset
* eval_dataset is the evaluation dataset
* tokenizer is the tokenizer used for the input data
* compute_metrics is the function used to compute the evaluation metrics


###  Evaluate the model

When we check the model perfomances from the training, both training and validation loss reduced with epochs. 

![alt text](image-1.png)

![alt text](image-2.png)

The accuracy got improved from 87% to 89% with five epochs and limited dataset.



