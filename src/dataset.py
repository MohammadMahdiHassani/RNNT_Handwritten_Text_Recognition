import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class IAMDataset(Dataset):
    def __init__(self, root_dir, split_file, transform=None, max_len=128):
        self.root_dir = root_dir
        self.transform = transform
        self.max_len = max_len
        self.data = []
        self.char_to_idx = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2}  # Basic vocab
        self.idx_to_char = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>'}

        # Load split file (e.g., trainset.txt from IAM dataset)
        with open(split_file, 'r') as f:
            lines = f.readlines()
        
        # Load IAM lines.txt for image-text pairs
        with open(os.path.join(root_dir, 'lines.txt'), 'r') as f:
            annotations = [line.strip().split() for line in f.readlines() if not line.startswith('#')]
        
        # Filter by split and build vocabulary
        img_ids = {line.strip() for line in lines}
        for anno in annotations:
            img_id = anno[0]
            if img_id in img_ids:
                text = ' '.join(anno[8:])  # Text is after 8th column
                img_path = os.path.join(root_dir, 'lines', f"{img_id}.png")
                if os.path.exists(img_path):
                    self.data.append((img_path, text))
                    # Update vocab
                    for char in text:
                        if char not in self.char_to_idx:
                            idx = len(self.char_to_idx)
                            self.char_to_idx[char] = idx
                            self.idx_to_char[idx] = char

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, text = self.data[idx]
        img = Image.open(img_path).convert('L')  # Grayscale
        
        if self.transform:
            img = self.transform(img)

        # Convert text to tensor
        text_tensor = [1] + [self.char_to_idx[char] for char in text] + [2]  # <SOS> + text + <EOS>
        text_tensor = text_tensor + [0] * (self.max_len - len(text_tensor))  # Pad
        text_tensor = torch.tensor(text_tensor[:self.max_len], dtype=torch.long)
        
        return img, text_tensor, len(text) + 2  # Length includes <SOS> and <EOS>

# Transform for preprocessing
transform = transforms.Compose([
    transforms.Resize((64, 256)),  # Height x Width
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])