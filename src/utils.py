import torch
def decode_prediction(logits, idx_to_char):
    # Greedy decoding for simplicity
    preds = torch.argmax(logits, dim=-1)
    text = []
    for pred in preds:
        chars = [idx_to_char[idx.item()] for idx in pred if idx.item() in idx_to_char]
        text.append(''.join(chars).replace('<PAD>', '').replace('<SOS>', '').replace('<EOS>', ''))
    return text