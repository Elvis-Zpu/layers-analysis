import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import numpy as np

# ---------------- Dataset Loader ----------------
class BreastUltrasoundDataset(Dataset):
    """Custom Dataset for Breast Ultrasound Images (original images only)."""
    def __init__(self, image_dir, transform=None, max_samples=None):
        self.image_dir = image_dir
        self.transform = transform
        self.class_names = ["normal", "benign", "malignant"]
        samples = []

        for idx, class_name in enumerate(self.class_names):
            class_folder = os.path.join(image_dir, class_name)
            image_files = [
                f for f in os.listdir(class_folder)
                if f.lower().endswith('.png') and "_mask" not in f.lower()
            ]
            image_files.sort()  # Consistent order
            for filename in image_files:
                path = os.path.join(class_folder, filename)
                samples.append((path, idx))

        if max_samples is not None:
            samples = samples[:max_samples]
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(label, dtype=torch.long)
        return image, label

# ---------------- Data Preparation Example ----------------
def prepare_dataloaders(image_root_dir, batch_size=4, val_ratio=0.2, manual_seed=42):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        )
    ])
    full_dataset = BreastUltrasoundDataset(image_root_dir, transform=preprocess)
    generator = torch.Generator().manual_seed(manual_seed)
    train_size = int((1.0 - val_ratio) * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=generator
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
    )

    print(f"Train/val split: {len(train_dataset)}/{len(val_dataset)}")
    return train_loader, val_loader, train_dataset, val_dataset

# ---------------- CLIP Fine-tuner ----------------
class CLIPFineTuner:
    """
    Fine-tuning class for CLIP image encoder on BUSI (or similar datasets).
    Only the classifier head is trained.
    """
    def __init__(self, clip_model, class_count, device, learning_rate=1e-3):
        self.device = device
        self.clip_model = clip_model.eval().to(device)
        for param in self.clip_model.parameters():
            param.requires_grad = False
        feature_dim = clip_model.visual.output_dim
        self.classifier = nn.Linear(feature_dim, class_count).to(device)
        self.optimizer = optim.Adam(self.classifier.parameters(), lr=learning_rate)
        self.loss_fn = nn.CrossEntropyLoss()

    def train_step(self, images, labels):
        self.optimizer.zero_grad()
        with torch.no_grad():
            image_features = self.clip_model.encode_image(images)
        logits = self.classifier(image_features)
        loss = self.loss_fn(logits, labels)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train_epoch(self, dataloader):
        self.clip_model.eval()
        self.classifier.train()
        total_loss, correct, total_samples, batch_count = 0, 0, 0, 0

        for images, labels in tqdm(dataloader, desc="Training"):
            images = images.to(self.device)
            labels = labels.to(self.device)
            loss = self.train_step(images, labels)
            total_loss += loss

            # For accuracy, eval classifier only
            with torch.no_grad():
                image_features = self.clip_model.encode_image(images)
                logits = self.classifier(image_features)
                predictions = logits.argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total_samples += labels.size(0)
            batch_count += 1

        avg_loss = total_loss / batch_count if batch_count else 0
        train_acc = 100.0 * correct / total_samples if total_samples else 0
        return avg_loss, train_acc

    def evaluate(self, dataloader):
        self.clip_model.eval()
        self.classifier.eval()
        correct, total, val_loss, batch_count = 0, 0, 0, 0

        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Evaluating"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                image_features = self.clip_model.encode_image(images)
                logits = self.classifier(image_features)
                predictions = logits.argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                val_loss += self.loss_fn(logits, labels).item()
                batch_count += 1

        accuracy = 100.0 * correct / total if total else 0
        avg_loss = val_loss / batch_count if batch_count else 0

        print(f"Validation accuracy: {accuracy:.2f}%, avg loss: {avg_loss:.4f}")
        return accuracy, avg_loss

# ---------------- Model Save/Load Utilities ----------------
def save_model(model, optimizer, epoch, path="saved_models/fine_tuned_clip_busi.pt"):
    """
    Save the classifier head and optimizer state for the fine-tuned CLIP model.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.classifier.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, path)
    print(f"Model saved at epoch {epoch + 1} to: {path}")

def load_fine_tuned_model(model, optimizer=None, path="saved_models/fine_tuned_clip_busi.pt", device='cpu'):
    """
    Load the classifier head and (optional) optimizer state for the fine-tuned CLIP model.
    Returns (model, optimizer, epoch) tuple if load succeeded, otherwise (None, None, None).
    """
    if not os.path.exists(path):
        print(f"No saved model found at: {path}. Ensure the path is correct.")
        return None, None, None

    checkpoint = torch.load(path, map_location=device)
    model.classifier.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint.get('epoch', None)

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Successfully loaded fine-tuned model and optimizer from: {path}")
    else:
        print(f"Successfully loaded fine-tuned model from: {path}")

    return model, optimizer, epoch

# ---------------- Example Gradient Tracking Utility ----------------
def track_layerwise_gradients(tuner, train_loader, val_loader, num_epochs=3):
    """
    Tracks mean absolute gradients per classifier parameter each epoch.
    Returns: layerwise_grads (np.ndarray)
    """
    layer_names = [name for name, _ in tuner.classifier.named_parameters()]
    num_layers = len(layer_names)
    layerwise_grads = np.zeros((num_layers, num_epochs))

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train_loss, train_acc = tuner.train_epoch(train_loader)

        epoch_grads = []
        for idx, (name, param) in enumerate(tuner.classifier.named_parameters()):
            grad_value = param.grad
            if grad_value is not None:
                mean_grad = grad_value.abs().mean().cpu().item()
            else:
                mean_grad = 0.0
            layerwise_grads[idx, epoch] = mean_grad
            epoch_grads.append(mean_grad)

        print("  Layerwise mean gradients:", {name: f"{val:.4e}" for name, val in zip(layer_names, epoch_grads)})
        val_acc, val_loss = tuner.evaluate(val_loader)

    print("\nCollected layerwise gradients matrix (shape {}):".format(layerwise_grads.shape))
    print("  Rows: ", layer_names)
    print("  Columns: epochs (per column)")
    print(layerwise_grads)
    return layerwise_grads