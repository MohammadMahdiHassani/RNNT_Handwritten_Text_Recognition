import torch
from torch.utils.data import DataLoader
from dataset import IAMWordsDataset, transform
from model import DTrOCR_RNNT
from loss import rnnt_loss
from tqdm import tqdm

def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    for imgs, targets, target_lengths in tqdm(train_loader):
        imgs, targets = imgs.to(device), targets.to(device)
        target_lengths = torch.tensor(target_lengths, dtype=torch.long).to(device)
        
        optimizer.zero_grad()
        logits = model(imgs, targets[:, :-1], target_lengths - 1)
        input_lengths = torch.full((imgs.size(0),), logits.size(1), dtype=torch.long).to(device)
        
        loss = rnnt_loss(logits, targets[:, 1:], input_lengths, target_lengths - 1)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for imgs, targets, target_lengths in tqdm(val_loader):
            imgs, targets = imgs.to(device), targets.to(device)
            target_lengths = torch.tensor(target_lengths, dtype=torch.long).to(device)
            
            logits = model(imgs, targets[:, :-1], target_lengths - 1)
            input_lengths = torch.full((imgs.size(0),), logits.size(1), dtype=torch.long).to(device)
            
            loss = rnnt_loss(logits, targets[:, 1:], input_lengths, target_lengths - 1)
            total_loss += loss.item()
    return total_loss / len(val_loader)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Dataset
    train_dataset = IAMWordsDataset(root_dir='iam_words', split_file='train.uttlist', transform=transform)
    val_dataset = IAMWordsDataset(root_dir='iam_words', split_file='validation.uttlist', transform=transform)
    test_dataset = IAMWordsDataset(root_dir='iam_words', split_file='test.uttlist', transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Model
    model = DTrOCR_RNNT(vocab_size=len(train_dataset.char_to_idx)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    epochs = 10
    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # Evaluate on test set
    test_loss = evaluate(model, test_loader, device)
    print(f"Test Loss: {test_loss:.4f}")
    
    torch.save(model.state_dict(), 'model.pth')