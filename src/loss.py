import torch

class RNNTLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, input_lengths, target_lengths):
        # Simplified RNNT loss (requires a proper implementation like warp-rnnt)
        # Here, we use a placeholder cross-entropy for demonstration
        B, T, L, V = logits.size()
        loss = torch.nn.functional.cross_entropy(logits.view(-1, V), targets.view(-1), reduction='sum')
        loss = loss / B
        ctx.save_for_backward(logits, targets, input_lengths, target_lengths)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        # Placeholder backward pass
        logits, targets, input_lengths, target_lengths = ctx.saved_tensors
        grad_logits = torch.zeros_like(logits)
        return grad_logits, None, None, None

def rnnt_loss(logits, targets, input_lengths, target_lengths):
    return RNNTLoss.apply(logits, targets, input_lengths, target_lengths)