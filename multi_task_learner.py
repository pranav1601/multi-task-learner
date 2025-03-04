import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
import numpy as np

class MultiTaskSentenceTransformer(nn.Module):
    def __init__(
        self, 
        model_name='bert-base-uncased',
        embedding_dim=768,
        pooling_strategy='mean',
        num_classes=5,  # For A: Sentence Classification
        ner_tags=9      # For B: Named Entity Recognition
    ):
        """
        Initialize the Multi-Task SentenceTransformer model
        """
        super(MultiTaskSentenceTransformer, self).__init__()
        
        # Load pre-trained transformer
        self.transformer = BertModel.from_pretrained(model_name)
        self.pooling_strategy = pooling_strategy
        
        # Shared projection layer
        if embedding_dim != self.transformer.config.hidden_size:
            self.projection = nn.Linear(self.transformer.config.hidden_size, embedding_dim)
            self.output_dim = embedding_dim
        else:
            self.projection = None
            self.output_dim = self.transformer.config.hidden_size
        
        # Layer normalization for the embeddings
        self.layer_norm = nn.LayerNorm(self.output_dim)
        
        # A: Sentence Classification Head
        self.classification_head = nn.Sequential(
            nn.Linear(self.output_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
        
        # B: Named Entity Recognition Head
        # NER requires token-level classification
        self.ner_head = nn.Sequential(
            nn.Linear(self.transformer.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, ner_tags)
        )
        
    def get_sentence_embedding(self, input_ids, attention_mask):
        """
        Get sentence embeddings using mean pooling 
        """
        # Get transformer outputs
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        
        # Mean pooling: average all token embeddings
        embeddings = (outputs.last_hidden_state * attention_mask.unsqueeze(-1)).sum(1)
        embeddings = embeddings / attention_mask.sum(-1, keepdim=True)
        
        # Apply projection if needed
        if self.projection is not None:
            embeddings = self.projection(embeddings)
        
        # Apply layer normalization
        embeddings = self.layer_norm(embeddings)
        
        return embeddings
    
    def forward(self, input_ids, attention_mask, task=None):
        """
        Forward pass through the model for the specified task
        """
        # Get transformer outputs (needed for all tasks)
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        
        results = {}
        
        # Sentence Embedding (shared between tasks)
        if task is None or task == 'embedding':
            sentence_embedding = self.get_sentence_embedding(input_ids, attention_mask)
            results['embedding'] = F.normalize(sentence_embedding, p=2, dim=1)
        
        # A: Sentence Classification
        if task is None or task == 'classification':
            # Use the pooled sentence embedding for classification
            if 'embedding' not in results:
                sentence_embedding = self.get_sentence_embedding(input_ids, attention_mask)
            else:
                sentence_embedding = results['embedding']
            
            # Unnormalize if needed for classification
            if task == 'classification':
                sentence_embedding = sentence_embedding * sentence_embedding.norm(dim=1, keepdim=True)
                
            classification_logits = self.classification_head(sentence_embedding)
            results['classification'] = classification_logits
        
        # B: Named Entity Recognition
        if task is None or task == 'ner':
            # Use token-level representations for NER
            ner_logits = self.ner_head(last_hidden_state)
            results['ner'] = ner_logits
        
        return results

# Example usage
def test_multi_task_model():
    # Initialize model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = MultiTaskSentenceTransformer(
        model_name='bert-base-uncased',
        embedding_dim=768,
        pooling_strategy='mean',
        num_classes=5,  # 5 sentence classes
        ner_tags=9      # 9 NER tags (B-PER, I-PER, B-ORG, etc.)
    )
    
    sample_sentences = [
        "I am excited to as a machine learning Apprentice.",
        "Fetch is a great place to work."
    ]
    
    # Tokenize sentences
    inputs = tokenizer(
        sample_sentences,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    
    # Forward pass
    results = model(inputs['input_ids'], inputs['attention_mask'])
    
    # Print results
    print("Multi-Task Model Results:")
    for task, output in results.items():
        print(f"Task: {task}, Output shape: {output.shape}")
    
    print("\nSentence Classification Logits (first sample):")
    print(results['classification'][0])
    
    print("\nNER Logits (first token of first sample):")
    print(results['ner'][0, 0])
    
    print("\nSentence Embedding (first 50 dimensions of first sample):")
    print(results['embedding'][0, :50])

if __name__ == "__main__":
    test_multi_task_model()