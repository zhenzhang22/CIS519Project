# CIS519Project
CIS519Project
roberta_base.ipynb has the various BERT base models.  
n_shots_bert_summary_text (1).ipynb has the BERT for sequence classification.  
setup_download_dataset.ipynb has the download and dataset setup.  
Run setup_download_dataset.ipynb first, and then run the rest of the files after the dataset got downloaded and cleaned up.

#The detailed introduction to "roberta_base.ipynb"  
This Python file is a script to train a sentiment analysis model on a dataset of product reviews using the RoBERTa model from the Hugging Face Transformers library. It first preprocesses the dataset by retaining only the necessary columns, converting scores to contextual labels (e.g., "very negative", "negative", "neutral", "positive", and "very positive"), cleaning the text by removing special characters, and filtering out comments with fewer than 20 words. The preprocessed dataset is then split into training, validation, and test sets.

The second part of this file focuses on the implementation of the RoBERTa model. Depending on the chosen feature, either the "Text" or "Summary" column is used for training. The tokenizer and model are instantiated based on the chosen pre-trained model, and the model is moved to the GPU if available. Various utility functions are defined to prepare input data, calculate evaluation metrics such as accuracy and weighted F1 score, and create custom PyTorch datasets and dataloaders for training, validation, and testing.

In the third part of the file, the RoBERTa model is trained for a specified number of epochs using the AdamW optimizer, linear learning rate schedule with warm-up, and CrossEntropyLoss as the loss function. During training, the loss and evaluation metrics are computed, and the best model is saved on the basis of validation accuracy. 

Finally, in the fourth part of the file, the saved model is evaluated on the test dataset. Accuracy and F1 score are reported as the main evaluation metrics, giving a comprehensive understanding of the model's performance in terms of both correct predictions and the balance between precision and recall.
