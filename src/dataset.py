# src/dataset.py
import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn_pad(batch):
    """
    Custom collate function for DataLoader.
    Handles variable-length ECG signals by padding them to the longest in the batch.
    
    Batch structure: [(signal_1, label_1), (signal_2, label_2), ...]
    """
    # Separate signals and labels
    signals = [item[0] for item in batch] # List of (12, T)
    labels = [item[1] for item in batch]
    
    # Transpose to (T, 12) because pad_sequence expects (Length, Dim)
    signals_T = [torch.tensor(s.T, dtype=torch.float32) for s in signals]
    
    # Pad sequences (batch_first=True -> (Batch, Max_T, 12))
    # padding_value=0.0 is standard for zero-padding
    signals_padded = pad_sequence(signals_T, batch_first=True, padding_value=0.0)
    
    # Transpose back to (Batch, 12, Max_T) for 1D-CNNs
    signals_padded = signals_padded.transpose(1, 2)
    
    # Create attention mask (1 for real data, 0 for padded)
    # This is critical for Transformers/Attention models
    lengths = torch.tensor([s.shape[0] for s in signals_T])
    mask = torch.arange(signals_padded.shape[2])[None, :] < lengths[:, None]
    
    labels_tensor = torch.stack([torch.tensor(l, dtype=torch.float32) for l in labels])
    
    return signals_padded, labels_tensor, mask