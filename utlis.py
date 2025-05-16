import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split
from collections import defaultdict, Counter
import random

# -----------------------------
# Step 1: SimCLR Augmentation
# -----------------------------
class SimCLRTransform:
    def __init__(self, image_size=32):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=3),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

    def __call__(self, x):
        xi = self.transform(x)
        xj = self.transform(x)
        return xi, xj

# -----------------------------
# Step 2: SimCLR Dataset Wrapper
# -----------------------------
class SimCLRDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.transform = SimCLRTransform()
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, idx):
        img, _ = self.base_dataset[idx]
        img_tensor = self.to_tensor(img)
        xi, xj = self.transform(img)
        return img_tensor, xi, xj

    def __len__(self):
        return len(self.base_dataset)

# -----------------------------
# Step 3: Stratified Sampler
# -----------------------------
def get_stratified_indices(dataset, samples_per_class, seed=42):
    label_to_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        label_to_indices[label].append(idx)
    random.seed(seed)
    return [i for label in range(10) for i in random.sample(label_to_indices[label], samples_per_class)]

def print_class_distribution(name, dataset, class_names):
    labels = [label for _, label in dataset]
    dist = Counter(labels)
    print(f"\n {name} Set Class Distribution:")
    for i, cls in enumerate(class_names):
        count = dist[i]
        percent = count / len(dataset) * 100
        print(f"{cls:>10}: {count} samples ({percent:.2f}%)")

# -----------------------------
# Step 4: Get Dataloaders
# -----------------------------
def get_simclr_and_labeled_dataloaders_with_stats(
    finetune_size=5000,
    val_ratio=0.2,
    prebatch_size=256,
    batch_size=256,
    num_workers=2,
    seed=42,
    print_stats=False
):
    torch.manual_seed(seed)
    random.seed(seed)

    # Transforms
    transform_finetune = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Load raw CIFAR-10 train
    raw_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=None)
    class_names = raw_dataset.classes

    # Get stratified fine-tuning indices (balanced)
    finetune_indices = get_stratified_indices(raw_dataset, samples_per_class=finetune_size // 10, seed=seed)
    pretrain_indices = list(set(range(len(raw_dataset))) - set(finetune_indices))

    # Pretrain dataset (no labels needed)
    pretrain_dataset = Subset(raw_dataset, pretrain_indices)
    simclr_dataset = SimCLRDataset(pretrain_dataset)
    simclr_loader = DataLoader(simclr_dataset, batch_size=prebatch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    if print_stats:
      print_class_distribution("Pretrain", pretrain_dataset, class_names)

    # Fine-tuning dataset
    labeled_dataset = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_finetune)
    finetune_subset = Subset(labeled_dataset, finetune_indices)
    num_val = int(val_ratio * finetune_size)
    train_subset, val_subset = random_split(finetune_subset, [finetune_size - num_val, num_val],
                                            generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if print_stats:
      print_class_distribution("Train", train_subset, class_names)
      print_class_distribution("Validation", val_subset, class_names)

    # Test set
    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_finetune)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    if print_stats:
      print_class_distribution("Test", test_set, class_names)

    return simclr_loader, train_loader, val_loader, test_loader



from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
import torch
import numpy as np

def evaluate(model, dataloader, device, num_classes=10, return_per_class=False):
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            all_logits.append(logits.cpu())
            all_labels.append(y.cpu())

    # Concatenate all batches
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    # Predicted class labels
    preds = all_logits.argmax(dim=1)
    acc = accuracy_score(all_labels, preds)

    # Convert to numpy
    y_true = all_labels.numpy()
    y_score = all_logits.numpy()

    # Binarize true labels for multi-class AUCs
    y_true_bin = np.eye(num_classes)[y_true]

    # Compute metrics
    try:
        roc_auc = roc_auc_score(y_true_bin, y_score, average="macro", multi_class="ovr")
    except ValueError:
        roc_auc = None  # Not computable with single-class batch

    try:
        pr_auc = average_precision_score(y_true_bin, y_score, average="macro")
    except ValueError:
        pr_auc = None

    #print(f"✅ Accuracy: {acc:.4f} | ROC AUC: {roc_auc:.4f} | PR AUC: {pr_auc:.4f}")

    if return_per_class:
        roc_auc_per_class = roc_auc_score(y_true_bin, y_score, average=None)
        pr_auc_per_class = average_precision_score(y_true_bin, y_score, average=None)
        return {
            "accuracy": acc,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "roc_auc_per_class": roc_auc_per_class,
            "pr_auc_per_class": pr_auc_per_class
        }

    return {
        "accuracy": acc,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc
    }


