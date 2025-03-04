# multi-task-learner
This repository shows how sentence transformers can be used for multi-task learning


## Architecture Choices Explanation - sentence transformers
* Base Model: I chose BERT (bert-base-uncased) as the transformer backbone because it provides strong performance for general NLP tasks. It has a solid understanding of language semantics, which is essential for generating meaningful sentence embeddings.

* Pooling Strategy: I chose to implement mean pooling and not CLS or max which averages all token embeddings, weighted by the attention mask. I preferred this as it considers all tokens equally.

* Projection Layer: I added an optional projection layer to adjust the embedding dimension if needed. This provides flexibility to match specific downstream task requirements.

* Layer Normalization: Applied to stabilize the embedding distributions, which helps with training stability and downstream task performance.

* L2 Normalization: I normalize the final embeddings to unit length, which is standard in sentence embedding tasks.

## Architecture Choices for Multi-Task Learning
* Shared Backbone: The BERT transformer backbone is shared between all tasks, allowing knowledge transfer across tasks and efficient parameter usage.

* Task-Specific Heads:
    * A (Sentence Classification): Added a classification head with a fully connected network that takes the sentence embedding and outputs class probabilities.
    * B (Named Entity Recognition): Added an NER head that operates at the token level, taking the token representations from the final hidden state and predicting entity tags for each token.


* Flexible Forward Pass: I implemented a task-specific forward pass that allows the model to:
    1. Generate sentence embeddings (useful for retrieval, similarity tasks)
    2. Perform sentence classification (Task A)
    3. Perform named entity recognition (Task B)
    4. Or all of the above simultaneously

* Shared Representation Learning: Both task heads build upon the same transformer representations, but use them differently:
    * Classification uses a mean pooled sentence-level representation
    * NER uses token-level representations

* Separation: Each task has a dedicated head with its own parameters, allowing for task-specific optimization while sharing the majority of parameters in the backbone.

