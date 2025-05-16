import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import csv
import time
import torchvision.models as models
from utils import *
from model import *
from loss import *

def pretrain_simclr(model, train_loader, loss_fn, optimizer, device, epochs=100):
    m = getattr(loss_fn, "m", 1)  # get block count if available
    batch_size = getattr(loss_fn, "batch_size", "unknown")
    csv_path = f"simclr_log_bs{batch_size}_m{m}.csv"

    model.to(device)
    model.train()
    logs = [("epoch", "avg_loss", "peak_memory_MB", "epoch_time_s")]

    for epoch in range(epochs):
        total_loss = 0
        torch.cuda.reset_peak_memory_stats()
        start_time = time.time()

        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for _, xi, xj in progress:
            xi, xj = xi.to(device), xj.to(device)

            _, zi = model(xi)
            _, zj = model(xj)

            loss = loss_fn(zi, zj)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        epoch_time = time.time() - start_time
        peak_mem = torch.cuda.max_memory_allocated(device=device) / 1024**2

        print(f"Epoch [{epoch+1}/{epochs}] - Avg Loss: {avg_loss:.4f} - Time: {epoch_time:.2f}s - Mem: {peak_mem:.1f}MB")
        logs.append((epoch+1, avg_loss, round(peak_mem, 1), round(epoch_time, 2)))

    torch.save(model.encoder.state_dict(), "simclr_encoder.pth")

    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(logs)

def train_linear_classifier_with_early_stopping(
    model, train_loader, val_loader, test_loader, device,
    epochs=50, patience=10, checkpoint_path="best_linear_model.pth", print_per_epoch = False
):
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)

    best_val_score = 0
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        # Training step
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Evaluate train and val
        train_metrics = evaluate(model, train_loader, device)
        val_metrics = evaluate(model, val_loader, device)
        val_score = val_metrics['accuracy']  # used for early stopping

        # Print epoch summary
        if print_per_epoch:
          print(f"\n Epoch [{epoch+1}/{epochs}]")
          print(f" Train — Acc: {train_metrics['accuracy']:.4f} | ROC AUC: {train_metrics['roc_auc']:.4f} | PR AUC: {train_metrics['pr_auc']:.4f}")
          print(f" Val   — Acc: {val_metrics['accuracy']:.4f} | ROC AUC: {val_metrics['roc_auc']:.4f} | PR AUC: {val_metrics['pr_auc']:.4f}")
          print(f"Loss: {total_loss:.4f}")

        # Save best model
        if val_score > best_val_score:
            best_val_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            if print_per_epoch:
              print(f" Saved new best model (val_acc: {val_metrics['accuracy']:.2f}%)")
        else:
            patience_counter += 1

        if patience_counter >= patience:
          if print_per_epoch:
            print(f"\n⏹️ Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"\n✅ Best Val Accuracy: {best_val_score:.4f} — loading best model from: {checkpoint_path}")

    # Final Test Evaluation
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    test_metrics = evaluate(model, test_loader, device)
    print(f"\n Final Test Results — "
          f"Accuracy: {test_metrics['accuracy']:.4f} | "
          f"ROC AUC: {test_metrics['roc_auc']:.4f} | "
          f"PR AUC: {test_metrics['pr_auc']:.4f}")



def build_linear_classifier(
    pretrained=False,
    base_model='resnet18',
    checkpoint_path=None,
    hidden_dim=512,
    num_classes=10,
    device='cuda'
):
    # Base encoder: ResNet18 (can be changed to ResNet34, etc.)
    #encoder = models.resnet18()
    encoder = getattr(models, base_model)(pretrained=False)
    encoder.fc = torch.nn.Identity()

    if pretrained:
        if checkpoint_path is None:
            raise ValueError("Checkpoint path required for pretrained encoder.")
        state_dict = torch.load(checkpoint_path, map_location=device)
        encoder.load_state_dict(state_dict, strict=False)
        print(f"✅ Loaded pretrained encoder from: {checkpoint_path}")
    else:
        print("⚠️ Using randomly initialized encoder.")

    # Combine with linear classifier
    model = LinearClassifier(encoder, hidden_dim=hidden_dim, num_classes=num_classes)
    return model



import os


def run_simclr_pipeline(
    base_model='resnet18',
    finetune_size=5000,
    val_ratio=0.2,
    prebatch_size=256,
    batch_size=256,
    m=1,
    num_workers=0,
    seed=42,
    pretrain_epochs=10,
    finetune_epochs=10,
    patience=10,
    projection_dim=128,
    pre_lr=1e-3,
    cl_temperature=0.5,
    pretrained_encoder=True,
    encoder_checkpoint_path="simclr_encoder.pth",
    classifier_checkpoint_path="best_linear_classifier.pth",
    extra_print=False,
    only_CLR=True,
    device='cuda'
):
    # Step 1: Load Data
    simclr_loader, train_loader, val_loader, test_loader = get_simclr_and_labeled_dataloaders_with_stats(
        finetune_size=finetune_size,
        val_ratio=val_ratio,
        batch_size=batch_size,
        prebatch_size=prebatch_size,
        num_workers=num_workers,
        seed=seed,
        print_stats=False
    )

    if device == None:
        device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"✅ Available device: {device}")
    else:
        device=device
        print(f"✅ Available device: {device}")

    # Step 2: Pretraining
    if pretrained_encoder:
        simclr_model = ResNetSimCLR(base_model=base_model, projection_dim=projection_dim)

        if os.path.exists(encoder_checkpoint_path):
            simclr_model.encoder.load_state_dict(torch.load(encoder_checkpoint_path, map_location=device))
            if extra_print:
                print(f"✅ Loaded existing pretrained encoder from: {encoder_checkpoint_path}")
        else:
            if extra_print:
                print(" Starting SimCLR Pretraining...")
            optimizer = optim.Adam(simclr_model.parameters(), lr=pre_lr)
            loss_fn = NTXentLoss(batch_size=prebatch_size, temperature=cl_temperature, m=m)
            pretrain_simclr(simclr_model, simclr_loader, loss_fn, optimizer, device, epochs=pretrain_epochs)
            torch.save(simclr_model.encoder.state_dict(), encoder_checkpoint_path)
            if extra_print:
                print(f" Pretrained encoder saved to: {encoder_checkpoint_path}")
    else:
        print("⚠️ Skipping SimCLR pretraining — using random initialization.")

    # Step 3: Fine-tuning (if not only_CLR)
    if not only_CLR:
        final_model = build_linear_classifier(
            pretrained=pretrained_encoder,
            base_model=base_model,
            checkpoint_path=encoder_checkpoint_path if pretrained_encoder else None,
            device=device
        )

        if extra_print:
            print(" Starting linear evaluation (fine-tuning)...")

        train_linear_classifier_with_early_stopping(
            final_model,
            train_loader,
            val_loader,
            test_loader,
            device,
            epochs=finetune_epochs,
            patience=patience,
            checkpoint_path=classifier_checkpoint_path
        )




