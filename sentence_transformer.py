import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
import numpy as np

class SentenceTransformer(nn.Module):
    def __init__(self, model_name='bert-base-uncased', embedding_dim=768):
        """
        Initialize the SentenceTransformer model
        """
        super(SentenceTransformer, self).__init__()
        
        self.transformer = BertModel.from_pretrained(model_name)
        
        # Add a projection layer to adjust embedding dimension if needed
        if embedding_dim != self.transformer.config.hidden_size:
            self.projection = nn.Linear(self.transformer.config.hidden_size, embedding_dim)
        else:
            self.projection = None
        
        # Layer normalization for the embeddings
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
    def forward(self, input_ids, attention_mask):
        """
        Forward pass through the model
        """
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        
        # Mean pooling: average all token embeddings
        embeddings = (outputs.last_hidden_state * attention_mask.unsqueeze(-1)).sum(1)
        embeddings = embeddings / attention_mask.sum(-1, keepdim=True)
        
        # Apply projection if needed
        if self.projection is not None:
            embeddings = self.projection(embeddings)
        
        # layer normalization
        embeddings = self.layer_norm(embeddings)
        
        # L2 normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings

def tokenize_sentences(sentences, tokenizer):
    """
    Tokenize a list of sentences
    """
    return tokenizer(
        sentences,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )

# Example usage
def get_sentence_embeddings(model, sentences, tokenizer, device='cpu'):
    """
    Get embeddings for a list of sentences
    """
    model.eval()
    model.to(device)
    
    with torch.no_grad():
        inputs = tokenize_sentences(sentences, tokenizer)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        
        embeddings = model(input_ids, attention_mask)
        
    return embeddings.cpu().numpy()

# Test with sample sentences
if __name__ == "__main__":
    # Initialize model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = SentenceTransformer(model_name='bert-base-uncased', embedding_dim=768)
    
    # Sample sentences
    sample_sentences = [
        "The cat sat on the mat.",
        "The dog barked at the mailman.",
        "I enjoy walking in the park on sunny days."
    ]
    
    # Get embeddings
    embeddings = get_sentence_embeddings(model, sample_sentences, tokenizer)
    
    # Print sample results
    for i, sentence in enumerate(sample_sentences):
        print(f"Sentence: {sentence}")
        print(f"Embedding shape: {embeddings[i].shape}")
        print(f"Embeddings (first 50): {embeddings[i][:50]}")
        print("-" * 50)
    