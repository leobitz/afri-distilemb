import torch
import torch.nn as nn

class FastTextModel(nn.Module):
    def __init__(self, file_path):
        super(FastTextModel, self).__init__()
        
        word2vec = {}
        # read word and vector pairs from the file
        vec_size = 512  # expected vector size
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    vec_size = int(parts[1])  # first line is the vector size
                    continue
                word = parts[0]
                try:

                    if len(parts[1:]) != vec_size:
                        raise ValueError(f"Vector for word '{word}' does not have the expected dimension of {vec_size}., found {len(word2vec[word])}.")
                    vector = list(map(float, parts[1:]))
                    word2vec[word] = torch.tensor(vector, dtype=torch.float32)
                except:
                    print(parts)
        self.word2id = {}
        self.word2id['<pad>'] = 0  # add pad token with ID 0
        # add unk token with a vector of zeros
        word2vec['<unk>'] = torch.zeros(vec_size, dtype=torch.float32)
        word2vec['<pad>'] = torch.zeros(vec_size, dtype=torch.float32)  # pad token vector
        # create an embedding layer with the size of the vocabulary and the dimension of the vectors
        self.embedding = nn.Embedding(len(word2vec) + 1, vec_size)
        # initialize the embedding weights with the vectors from the file
        weights = torch.zeros(len(word2vec), vec_size, dtype=torch.float32)
        
        for word, vec in word2vec.items():
            if word in self.word2id:
                continue
            self.word2id[word] = len(self.word2id)  # assign a unique ID to each word
            weights[self.word2id[word]] = vec
        self.embedding.weight = nn.Parameter(weights, requires_grad=False)  # freeze the weights


    def forward(self, input_ids, attention_mask, pool=False, **kwargs):
        """
        Forward pass of the model.
        
        Args:
            input_ids (torch.Tensor): Input tensor containing token IDs.
        
        Returns:
            torch.Tensor: Output tensor after passing through the model.
        """
        embeddings = self.embedding(input_ids)
        if pool:
            # Apply attention mask: set masked positions to 0
            mask = attention_mask.unsqueeze(-1).float()
            masked_embeddings = embeddings * mask
            # Avoid division by zero
            lengths = mask.sum(dim=1).clamp(min=1)
            pooled = masked_embeddings.sum(dim=1) / lengths
            return pooled
        else:
            return embeddings
