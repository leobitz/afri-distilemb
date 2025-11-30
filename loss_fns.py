from typing import Optional
import torch
import torch.nn.functional as F


def generate_similars_from_embeddings(embeddings, top_k=1, mask=None):
    # embeddings.shape = (b, s, d)
    # mask.shape = (b, s) or None
    batch_size, seq_len, dim = embeddings.shape
    # Normalize embeddings for cosine similarity calculation
    embeddings = F.normalize(embeddings, p=2, dim=-1)
    
    # Calculate cosine similarity matrix for each item in the batch
    # (b, s, d) @ (b, d, s) -> (b, s, s)
    similarity_matrix = torch.bmm(embeddings, embeddings.transpose(1, 2))
    
    # Mask out the diagonal to avoid selecting the same embedding
    identity = torch.eye(seq_len, device=embeddings.device, dtype=torch.bool).unsqueeze(0)
    similarity_matrix.masked_fill_(identity, -float('inf'))
    # Mask out similarities that are too high (close to 1), indicating duplicates
    similarity_matrix.masked_fill_(similarity_matrix > 0.9999, -float('inf'))
    if mask is not None:
        # If a mask is provided, apply it to the similarity matrix
        # mask.shape = (b, s)
        mask = mask.unsqueeze(1).expand(-1, seq_len, -1)
        similarity_matrix.masked_fill_(~mask.bool(), -float('inf'))
    # similarity_matrix.shape = (b, s, s)
    # Find the top_k most similar embeddings
    top_k_similarities, top_k_indices = torch.topk(similarity_matrix, k=top_k, dim=-1)
    
    # top_k_indices.shape = (b, s, top_k)
    # We need to gather the actual embeddings.
    # To do this, we'll expand top_k_indices to match the embedding dimension.
    # expanded_indices shape: (b, s, top_k, d)
    expanded_indices = top_k_indices.unsqueeze(-1).expand(-1, -1, -1, dim)
    
    # We need to reshape embeddings to be able to gather from it using the indices.
    # embeddings shape: (b, s, d) -> (b, 1, s, d) -> (b, s, s, d)
    # This allows us to select from the original sequence for each item.
    expanded_embeddings = embeddings.unsqueeze(1).expand(-1, seq_len, -1, -1)

    # Gather the top_k embeddings
    # similar_embeddings.shape = (b, s, top_k, d)
    similar_embeddings = torch.gather(expanded_embeddings, 2, expanded_indices)
    
    if top_k == 1:
        return similar_embeddings.squeeze(2) # (b, s, d)
    return similar_embeddings # (b, s, top_k, d)

def info_nce_loss(anchor, positive, negatives=None, temperature=0.1):
    # anchor.shape = (b, d)
    # positive.shape = (b, d)
    # negatives.shape = (b, n, d) or None
    if anchor.dim() != 2 or positive.dim() != 2:
        raise ValueError("Anchor and positive must be 2D tensors with shape (b, d).")
    
    if negatives is None:
        negatives = generate_similars_from_embeddings(positive.unsqueeze(0), 3).squeeze(0)
    
    # Compute similarities
    sim_positive = F.cosine_similarity(anchor, positive, dim=-1)  # Similarity with positive
    sim_negatives = F.cosine_similarity(anchor.unsqueeze(1), negatives, dim=-1)  # Similarities with negatives
    
    # Scale by temperature
    sim_positive = sim_positive / temperature
    sim_negatives = sim_negatives / temperature
    
    # Exponentiate similarities
    exp_positive = torch.exp(sim_positive)
    exp_negatives = torch.exp(sim_negatives)
    
    # Compute denominator
    denominator = exp_positive + torch.sum(exp_negatives, dim=-1)
    
    # Compute InfoNCE loss
    loss = -torch.log(exp_positive / denominator)
    return loss.mean()  # Mean over batch


def info_nce_loss_v2(anchor, positive, negatives, attention_mask=None, temperature=0.1):
    # anchor.shape = (b, s, d)
    # positive.shape = (b, s, d)
    # negatives.shape = (b, s, n, d)
    # attention_mask.shape = (b, s)
    
    # Compute similarities
    # sim_positive.shape = (b, s)
    sim_positive = F.cosine_similarity(anchor, positive, dim=-1)
    # anchor.unsqueeze(2).shape = (b, s, 1, d)
    # negatives.shape = (b, s, n, d)
    # sim_negatives.shape = (b, s, n)
    sim_negatives = F.cosine_similarity(anchor.unsqueeze(2), negatives, dim=-1)
    
    # Scale by temperature
    sim_positive = sim_positive / temperature
    sim_negatives = sim_negatives / temperature
    
    # Exponentiate similarities
    exp_positive = torch.exp(sim_positive)
    exp_negatives = torch.exp(sim_negatives)
    
    # Compute denominator
    # sum over negative samples, shape: (b, s)
    sum_exp_negatives = torch.sum(exp_negatives, dim=-1)
    denominator = exp_positive + sum_exp_negatives
    
    # Compute InfoNCE loss
    # loss per token, shape: (b, s)
    loss = -torch.log(exp_positive / denominator)
    
    if attention_mask is not None:
        # Apply mask to the loss
        loss = loss.masked_fill(~attention_mask.bool(), 0)
        # Compute mean loss only over non-masked tokens
        masked_loss = loss.sum() / attention_mask.sum()
        return masked_loss
    
    return loss.mean()