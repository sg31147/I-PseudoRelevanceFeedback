import torch


def f1_score_db_tuning(logits, targets, average="micro", type="single"):
    # Automatically select GPU if available, otherwise use CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if average not in ["micro", "macro"]:
        raise ValueError("Average must be either 'micro' or 'macro'")
    
    # Move tensors to the selected device (GPU if available, otherwise CPU)
    logits = logits.to(device)
    targets = targets.to(device)

    dbs = torch.linspace(0, 1, 100, device=device)  # Ensure linspace is created on the correct device
    tp = torch.zeros((len(dbs), targets.shape[1]), device=device)  # Initialize tensors on GPU or CPU
    fp = torch.zeros((len(dbs), targets.shape[1]), device=device)
    fn = torch.zeros((len(dbs), targets.shape[1]), device=device)
    
    for idx, db in enumerate(dbs):
        predictions = (logits > db).long()
        predictions
        tp[idx] = torch.sum((predictions) * (targets), dim=0)
        fp[idx] = torch.sum(predictions * (1 - targets), dim=0)
        fn[idx] = torch.sum((1 - predictions) * targets, dim=0)
    
    if average == "micro":
        f1_scores = tp.sum(1) / (tp.sum(1) + 0.5 * (fp.sum(1) + fn.sum(1)) + 1e-10)
    else:
        f1_scores = torch.mean(tp / (tp + 0.5 * (fp + fn) + 1e-10), dim=1)

    # After the computations, bring results back to the CPU only once
    f1_scores = f1_scores.cpu()
    dbs = dbs.cpu()

    if type == "single": #เก็บค่าเดียว 
        best_f1 = f1_scores.max()
        best_db = dbs[f1_scores.argmax()]
        print(f"Best F1: {best_f1:.4f} at DB: {best_db:.4f}")
        return best_f1, best_db
    if type == "per_class":
        best_f1 = f1_scores.max(1)
        best_db = dbs[f1_scores.argmax(0)]
        print(f"Best F1: {best_f1} at DB: {best_db}")
        return best_f1, best_db # คืน คืน f1 กับ threshold ที่ดีที่สุด

def f1_score_db_tuning_chunked(logits, targets, average="micro", type="single", chunk_size=25):
    # Automatically select GPU if available, otherwise use CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if average not in ["micro", "macro"]:
        raise ValueError("Average must be either 'micro' or 'macro'")

    # Move tensors to the selected device
    logits = logits.to(device)
    targets = targets.to(device)

    dbs = torch.linspace(0, 1, 3, device=device)  # Thresholds

    # Initialize tensors to store F1 scores
    f1_scores = torch.empty(len(dbs), device=device)

    # Chunked computation to reduce memory usage
    for start in range(0, len(dbs), chunk_size):
        end = min(start + chunk_size, len(dbs))
        db_chunk = dbs[start:end].view(-1, 1, 1)  # Shape: (chunk_size, 1, 1)

        # Broadcasting for thresholding
        predictions = (logits.unsqueeze(0) > db_chunk).long()  # Shape: (chunk_size, batch_size, num_classes)

        tp = torch.sum(predictions * targets.unsqueeze(0), dim=1)
        fp = torch.sum(predictions * (1 - targets.unsqueeze(0)), dim=1)
        fn = torch.sum((1 - predictions) * targets.unsqueeze(0), dim=1)
       
        if average == "micro":
            tp_sum = tp.sum(dim=1)  # Sum over classes
            fp_sum = fp.sum(dim=1)
            fn_sum = fn.sum(dim=1)
            f1_scores[start:end] = tp_sum / (tp_sum + 0.5 * (fp_sum + fn_sum) + 1e-10)
        else:
            f1_scores[start:end] = torch.mean(tp / (tp + 0.5 * (fp + fn) + 1e-10), dim=1)

    # Bring results back to the CPU only once
    f1_scores = f1_scores.cpu()
    dbs = dbs.cpu()

    if type == "single":
        best_f1 = f1_scores.max()
        best_db = dbs[f1_scores.argmax()]
        print(f"Best F1: {best_f1:.4f} at DB: {best_db:.4f}")
        return best_f1, best_db
    elif type == "per_class":
        best_f1 = f1_scores.max(1)
        best_db = dbs[f1_scores.argmax(0)]
        print(f"Best F1: {best_f1} at DB: {best_db}")
        return best_f1, best_db





