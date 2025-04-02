import torch


def pseudo_relevance_feedback(batch_a, batch_b,CosSim_Thresh = 0.00, TopKSelection=5,alpha=1,beta=0.1,gramma=0, chunk_size_b=10000):

    epsilon = 1e-15
 
    relevance_feedback_result = torch.zeros_like(batch_b)  # Result tensor must match the shape of batch_b
    
    a_norm = torch.linalg.norm(batch_a, dim=1, keepdim=True)+epsilon

    
    # Iterate over chunks of batch_b
    for j in range(0, batch_b.size(0), chunk_size_b):
        
        b_chunk = batch_b[j:j + chunk_size_b]
        
        b_norm = torch.linalg.norm(b_chunk, dim=1, keepdim=True)+epsilon
      
        # Compute dot product and cosine similarity
        dot_product = torch.matmul(batch_a, b_chunk.T)
        norm_matrix = a_norm * b_norm.T
        cosine_similarity = (dot_product / norm_matrix).T
        
        topk_values, topk_indices = torch.topk(cosine_similarity, TopKSelection, dim=1)

        # Masks: Above median -> Relevant, Below median -> Non-relevant
        relevant_mask = topk_values >= CosSim_Thresh  # (2x5)
        non_relevant_mask = topk_values < CosSim_Thresh  # (2x5)
        
        relevant_vecs=remove_trailing_zeros(batch_a[topk_indices]*relevant_mask.unsqueeze(2))
        non_relevant_vecs=remove_trailing_zeros(batch_a[topk_indices]*non_relevant_mask.unsqueeze(2))
        
        

        relevance_doc = torch.stack([t.mean(dim=0, keepdim=True) for t in relevant_vecs]).squeeze(1)
        non_relevance_doc = torch.stack([t.mean(dim=0, keepdim=True) for t in non_relevant_vecs]).squeeze(1)
        
        if torch.isnan(relevance_doc).any():
            relevance_doc=torch.zeros(b_chunk.size(0), b_chunk.size(1))
           
        if torch.isnan(non_relevance_doc).any():
            non_relevance_doc=torch.zeros(b_chunk.size(0), b_chunk.size(1))
        
        relevance_feedback_result[j:j + chunk_size_b] = ((alpha*b_chunk) + (beta * relevance_doc) - (gramma* non_relevance_doc))

      
    return relevance_feedback_result
     
def remove_trailing_zeros(tensor):
    mask = ~(tensor == 0).all(dim=-1)  # Identify non-zero rows
    return [t[mask[i]] for i, t in enumerate(tensor)]