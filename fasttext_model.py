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
                    vector = list(map(float, parts[1:]))
                    word2vec[word] = torch.tensor(vector, dtype=torch.float32)
                    if len(word2vec[word]) != vec_size:
                        raise ValueError(f"Vector for word '{word}' does not have the expected dimension of {vec_size}., found {len(word2vec[word])}.")
                except:
                    print(parts)
        # add unk token with a vector of zeros
        unk_vector = torch.zeros(vec_size, dtype=torch.float32)
        word2vec['<unk>'] = unk_vector
        # create an embedding layer with the size of the vocabulary and the dimension of the vectors
        self.embedding = nn.Embedding(len(word2vec), vec_size)
        # initialize the embedding weights with the vectors from the file
        weights = torch.zeros(len(word2vec), vec_size, dtype=torch.float32)
        self.word2id = {}
        for word, vec in weights.items():
            self.word2id[word] = len(self.word2id)  # assign a unique ID to each word
            weights[self.word2id[word]] = vec
        self.embedding.weight = nn.Parameter(weights, requires_grad=False)  # freeze the weights


    def forward(self, input_ids, **kwargs):
        """
        Forward pass of the model.
        
        Args:
            input_ids (torch.Tensor): Input tensor containing token IDs.
        
        Returns:
            torch.Tensor: Output tensor after passing through the model.
        """
        assert input_ids.dim() == 2, "Input tensor must be 2D (batch_size, seq_length)"
        # if input_id is not in the vocabulary, replace it with the unk token
        embedded = self.embedding(input_ids)
        # Global average pooling
        pooled = embedded.mean(dim=1)
        return pooled

