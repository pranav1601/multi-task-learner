import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultiTaskDataset(Dataset):
    def __init__(self, texts, classification_labels=None, ner_labels=None, tokenizer=None, max_length=128):
        """Simple dataset for multi-task learning"""
        self.texts = texts
        self.classification_labels = classification_labels
        self.ner_labels = ner_labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
        }
        
        if self.classification_labels is not None:
            item['classification_label'] = torch.tensor(self.classification_labels[idx])
        
        if self.ner_labels is not None:
            ner_label = self.ner_labels[idx]
            padded_labels = torch.full((self.max_length,), -100)  # -100 is ignored in loss
            padded_labels[:min(len(ner_label), self.max_length)] = torch.tensor(ner_label[:self.max_length])
            item['ner_labels'] = padded_labels
        
        return item

class MultiTaskTrainer:
    def __init__(
        self,
        model,
        train_dataset,
        val_dataset=None,
        batch_size=16,
        learning_rate=2e-5,
        num_epochs=3,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        task_weights={'classification': 1.0, 'ner': 1.0},
        progressive_unfreeze=True,
        unfreeze_epoch=1
    ):
        """
        trainer for multi-task learning
        """
        self.model = model
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.device = device
        self.task_weights = task_weights
        self.progressive_unfreeze = progressive_unfreeze
        self.unfreeze_epoch = unfreeze_epoch
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2
        )
        
        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=2
            )
        else:
            self.val_loader = None
        
        self.model.to(device)
        
        # Initially freeze the backbone
        if self.progressive_unfreeze:
            self._freeze_backbone()
        
        # Set up optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        
        
        total_steps = len(self.train_loader) * num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
        
        
        self.classification_loss_fn = nn.CrossEntropyLoss()
        self.ner_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    
    def _freeze_backbone(self):
        """Freeze the transformer backbone of the model"""
        for param in self.model.transformer.parameters():
            param.requires_grad = False
        
        logger.info("Transformer backbone frozen")
    
    def _unfreeze_layers(self, num_layers):
        """Unfreeze a specific number of transformer layers from the top"""
        self._freeze_backbone()
        
        total_layers = len(self.model.transformer.encoder.layer)
        
        layers_to_unfreeze = min(num_layers, total_layers)
        
        logger.info(f"Unfreezing top {layers_to_unfreeze} of {total_layers} layers")
        
        # Unfreeze layers from the top (last layers first)
        for i in range(total_layers - layers_to_unfreeze, total_layers):
            for param in self.model.transformer.encoder.layer[i].parameters():
                param.requires_grad = True
    
    def train(self):
        """Train the model"""
        logger.info("Starting training...")
        
        best_val_score = 0.0
        
        for epoch in range(self.num_epochs):
            logger.info(f"Epoch {epoch+1}/{self.num_epochs}")
            
            # Progressive unfreezing
            if self.progressive_unfreeze and epoch >= self.unfreeze_epoch:
                # Calculate how many layers to unfreeze: more layers for later epochs
                layers_to_unfreeze = 3 * (epoch - self.unfreeze_epoch + 1)
                self._unfreeze_layers(layers_to_unfreeze)
            
            self.model.train()
            train_loss = 0.0
            train_steps = 0
            
            progress_bar = tqdm(self.train_loader, desc=f"Training epoch {epoch+1}")
            
            for batch in progress_bar:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                
                loss = 0.0
                
                # Classification loss
                if 'classification_label' in batch and 'classification' in outputs:
                    cls_loss = self.classification_loss_fn(
                        outputs['classification'], 
                        batch['classification_label']
                    )
                    loss += self.task_weights['classification'] * cls_loss
                
                # NER loss
                if 'ner_labels' in batch and 'ner' in outputs:
                    batch_size, seq_len, num_labels = outputs['ner'].shape
                    ner_loss = self.ner_loss_fn(
                        outputs['ner'].view(-1, num_labels),
                        batch['ner_labels'].view(-1)
                    )
                    loss += self.task_weights['ner'] * ner_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.scheduler.step()
                
                train_loss += loss.item()
                train_steps += 1
                progress_bar.set_postfix({'loss': loss.item()})
            
            # Calculate average training loss
            avg_train_loss = train_loss / train_steps
            logger.info(f"Average training loss: {avg_train_loss:.4f}")
            
            # Validation phase
            if self.val_loader:
                val_metrics = self.evaluate()
                logger.info(f"Validation metrics: {val_metrics}")
                
                combined_score = 0.0
                if 'classification_accuracy' in val_metrics:
                    combined_score += val_metrics['classification_accuracy']
                if 'ner_f1' in val_metrics:
                    combined_score += val_metrics['ner_f1']
                
                num_tasks = sum(1 for k in val_metrics if k in ['classification_accuracy', 'ner_f1'])
                if num_tasks > 0:
                    combined_score /= num_tasks
                
                logger.info(f"Combined validation score: {combined_score:.4f}")
                
                # Save model if improved
                if combined_score > best_val_score:
                    best_val_score = combined_score
                    torch.save(self.model.state_dict(), f"model_epoch_{epoch+1}.pt")
                    logger.info(f"Model saved with score: {combined_score:.4f}")
            
            logger.info("-" * 50)
        
        logger.info("Training complete!")
        return best_val_score
    
    def evaluate(self):
        """Evaluate the model on the validation dataset"""
        self.model.eval()
        
        # Track predictions and labels
        all_cls_preds = []
        all_cls_labels = []
        all_ner_preds = []
        all_ner_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                
                # Collect classification predictions and labels
                if 'classification_label' in batch and 'classification' in outputs:
                    preds = torch.argmax(outputs['classification'], dim=1)
                    all_cls_preds.extend(preds.cpu().numpy())
                    all_cls_labels.extend(batch['classification_label'].cpu().numpy())
                
                # Collect NER predictions and labels
                if 'ner_labels' in batch and 'ner' in outputs:
                    # Get predictions
                    preds = torch.argmax(outputs['ner'], dim=2)
                    
                    # Only consider tokens that are not padding (-100)
                    for i in range(len(batch['ner_labels'])):
                        mask = batch['ner_labels'][i] != -100
                        all_ner_preds.extend(preds[i][mask].cpu().numpy())
                        all_ner_labels.extend(batch['ner_labels'][i][mask].cpu().numpy())
        
        # Calculate metrics
        metrics = {}
        
        # Classification metrics
        if all_cls_preds:
            metrics['classification_accuracy'] = accuracy_score(all_cls_labels, all_cls_preds)
            metrics['classification_f1'] = f1_score(all_cls_labels, all_cls_preds, average='macro')
        
        # NER metrics
        if all_ner_preds:
            metrics['ner_accuracy'] = accuracy_score(all_ner_labels, all_ner_preds)
            metrics['ner_f1'] = f1_score(all_ner_labels, all_ner_preds, average='macro')
        
        return metrics


# Example usage
def run_example():
    from transformers import BertTokenizer
    from multi_task_model import MultiTaskSentenceTransformer
    
    # Create sample data
    texts = [
        "Apple is looking at buying U.K. startup for $1 billion.",
        "John Smith lives in New York and works at Google.",
        "Scientists discover a new species in the Amazon rainforest."
    ]
    
    # Classification labels (0=Tech, 1=Person, 2=Science)
    classification_labels = [0, 1, 2]
    
    # Simple NER labels (0=O, 1=ORG, 2=PER, 3=LOC)
    ner_labels = [
        [0, 1, 0, 0, 0, 0, 3, 0, 0, 0, 0],
        [0, 2, 2, 0, 0, 3, 3, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 3, 0]
    ]
    
    # Create tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = MultiTaskSentenceTransformer(
        model_name='bert-base-uncased',
        num_classes=3,
        ner_tags=4
    )
    
    # Create datasets
    train_dataset = MultiTaskDataset(
        texts=texts,
        classification_labels=classification_labels,
        ner_labels=ner_labels,
        tokenizer=tokenizer
    )
    
    # For demonstration, using the same data for validation
    val_dataset = MultiTaskDataset(
        texts=texts,
        classification_labels=classification_labels,
        ner_labels=ner_labels,
        tokenizer=tokenizer
    )
    
    # Create trainer
    trainer = MultiTaskTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=2,
        learning_rate=1e-5,
        num_epochs=3,
        progressive_unfreeze=True,
        unfreeze_epoch=1
    )
    
    # Run training
    trainer.train()


if __name__ == "__main__":
    run_example()