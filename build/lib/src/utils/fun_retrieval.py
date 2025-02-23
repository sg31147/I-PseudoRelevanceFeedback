import torch

# Define the function
def prf(batch_a, batch_b, AvgTopR=40, AvgLowR=10,w_alpha=1,w_beta=0.1,w_gramma=0.1, chunk_size_b=80000):

    # batch_a represents the retrieval data (in this research, the self-training dataset) 
    # with dimensions RR x l.
    # batch_b represents the predicted output of the testing dataset from the model,
    # with dimensions RT x l.
    # NavgTop defines the number of relevant documents to retrieve.
    # NavgLow defines the number of irrelevant documents to exclude.
    # w_alpha specifies the term weight that pseudo-relevance feedback will prioritize.
    # w_beta specifies the term weight that pseudo-relevance feedback will de-emphasize.
    # chunk_size_b defines the batch size for dividing the testing dataset.It is too large to process at once.

    epsilon = 1e-21
 
    rag_result = torch.zeros_like(batch_b)  # Result tensor must match the shape of batch_b
    
    a_norm = torch.linalg.norm(batch_a, dim=1, keepdim=True)+epsilon


    # Iterate over chunks of batch_b
    for j in range(0, batch_b.size(0), chunk_size_b):
        
        b_chunk = batch_b[j:j + chunk_size_b]
   
        b_norm = torch.linalg.norm(b_chunk, dim=1, keepdim=True)+epsilon
    
        # Compute dot product and cosine similarity
        dot_product = torch.matmul(batch_a, b_chunk.T)
        norm_matrix = a_norm * b_norm.T
        cosine_similarity = dot_product / norm_matrix

        # Handle NavgTop and NavgLow
        if AvgTopR > 0:
            topk_values, topk_indices = torch.topk(cosine_similarity, AvgTopR, dim=0)
            top_mean_vector = batch_a[topk_indices.T].mean(dim=1)
        else:
            top_mean_vector = torch.zeros(b_chunk.size(0), batch_a.size(1))

        if AvgLowR > 0:
                lowk_values, lowk_indices = torch.topk(-cosine_similarity, AvgLowR, dim=0)
                low_mean_vector = batch_a[lowk_indices.T].mean(dim=1)
        else:
            low_mean_vector = torch.zeros(b_chunk.size(0), batch_a.size(1))

        rag_result[j:j + chunk_size_b] = (w_alpha*b_chunk) + (w_beta * top_mean_vector) + (w_gramma* -low_mean_vector)
  
    return rag_result


