# multi-task-learner
This repository shows how sentence transformers can be used for multi-task learning


## Architecture Choices Explanation - sentence transformers
* Base Model: I chose BERT (bert-base-uncased) as the transformer backbone because it provides strong performance for general NLP tasks. It has a solid understanding of language semantics, which is essential for generating meaningful sentence embeddings.

* Pooling Strategy: I chose to implement mean pooling and not CLS or max which averages all token embeddings, weighted by the attention mask. I preferred this as it considers all tokens equally.

* Projection Layer: I added an optional projection layer to adjust the embedding dimension if needed. This provides flexibility to match specific downstream task requirements.

* Layer Normalization: Applied to stabilize the embedding distributions, which helps with training stability and downstream task performance.

* L2 Normalization: I normalize the final embeddings to unit length, which is standard in sentence embedding tasks.