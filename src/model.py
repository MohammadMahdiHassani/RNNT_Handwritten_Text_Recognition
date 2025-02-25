import torch
import torch.nn as nn
from torchvision.models import resnet18

class DTrOCR_RNNT(nn.Module):
    def __init__(self, vocab_size, hidden_size=256, num_layers=2):
        super(DTrOCR_RNNT, self).__init__()
        
        # Encoder (DTrOCR-inspired Transformer-based feature extractor)
        self.cnn = resnet18(pretrained=True)
        self.cnn.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)  # Grayscale input
        self.cnn.fc = nn.Identity()  # Remove classification head
        
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8),
            num_layers=4
        )
        
        # Prediction Network (RNN for RNNT)
        self.pred_net = nn.LSTM(vocab_size, hidden_size, num_layers, batch_first=True)
        
        # Joint Network (RNNT)
        self.joint_net = nn.Sequential(
            nn.Linear(512 + hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, vocab_size)
        )
        
        self.vocab_size = vocab_size

    def forward(self, x, target=None, target_lengths=None):
        # CNN feature extraction
        batch_size = x.size(0)
        features = self.cnn(x)  # [B, 512, H', W']
        features = features.flatten(2).permute(0, 2, 1)  # [B, T, 512]
        
        # Transformer Encoder
        enc_output = self.encoder(features)  # [B, T, 512]
        
        if target is None:  # Inference mode
            return enc_output
        
        # Prediction Network
        target_onehot = nn.functional.one_hot(target, self.vocab_size).float()  # [B, L, V]
        pred_output, _ = self.pred_net(target_onehot)  # [B, L, hidden_size]
        
        # Joint Network
        T, L = enc_output.size(1), pred_output.size(1)
        enc_output = enc_output.unsqueeze(2).repeat(1, 1, L, 1)  # [B, T, L, 512]
        pred_output = pred_output.unsqueeze(1).repeat(1, T, 1, 1)  # [B, T, L, hidden]
        joint_input = torch.cat([enc_output, pred_output], dim=-1)  # [B, T, L, 512 + hidden]
        logits = self.joint_net(joint_input)  # [B, T, L, vocab_size]
        
        return logits
