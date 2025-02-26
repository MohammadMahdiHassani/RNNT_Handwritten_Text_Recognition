import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from pathlib import PosixPath
import torchvision.transforms as transforms
from pathlib import Path
class Word:
    def __init__(self, word_id, file_path, writer_id, transcription):
        self.id:str = word_id
        self.file_path:Path = file_path
        self.writer_id:str = writer_id
        self.transcription:str = transcription

    def __repr__(self):
        return (f"Word(id='{self.id}', file_path=PosixPath('{self.file_path}'), "
                f"writer_id='{self.writer_id}', transcription='{self.transcription}')")
class IAMWordsDataset(Dataset):
    def __init__(self, root_dir, split_file, transform=None, max_len=32):
        self.root_dir = root_dir
        self.transform = transform
        self.max_len = max_len
        self.data = []
        self.char_to_idx = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2}
        self.idx_to_char = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>'}

        # Load split file (e.g., train.uttlist)
        with open(os.path.join(root_dir, 'splits', split_file), 'r') as f:
            split_ids = {line.strip() for line in f.readlines()}

        # Simulate loading your word list (replace with your actual list loading logic)
        # For this example, I'll assume you pass the word list directly or load it here
        words_list = self._load_words_list()  # Replace with your actual word list loading

        # Filter by split and build dataset
        for word in words_list:
            if word.id in split_ids and os.path.exists(word.file_path):
                self.data.append((word.file_path, word.transcription))
                for char in word.transcription:
                    if char not in self.char_to_idx:
                        idx = len(self.char_to_idx)
                        self.char_to_idx[char] = idx
                        self.idx_to_char[idx] = char

    def _load_words_list(self):
        # Placeholder: Replace with your actual logic to load the words list
        # Example based on your format
        sample_words = [
            Word('a01-000u', PosixPath('/content/iam_words/words/a01/a01-000u/a01-000u-00-00.png'), '000', 'A'),
            Word('a01-000u', PosixPath('/content/iam_words/words/a01/a01-000u/a01-000u-00-01.png'), '000', 'MOVE'),
            Word('a01-000u', PosixPath('/content/iam_words/words/a01/a01-000u/a01-000u-00-02.png'), '000', 'to'),
            # Add your full list loading logic here
        ]
        return sample_words  # Replace with real data loading

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
    transforms.Resize((32, 128)),  # Smaller size for words (adjust as needed)
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])