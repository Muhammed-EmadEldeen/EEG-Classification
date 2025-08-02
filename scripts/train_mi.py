import torch
import torch.nn as nn
import torch.optim as optim
from models.model_mi import GRUClassifier
from scripts.load_mi_data import mi_loaders
from scripts.evaluate_mi import mi_evaluate


model = GRUClassifier()
max_acc = 0
train_loader, val_loader = mi_loaders()

def train(model, train_loader, val_loader, num_epochs=10, lr=1e-3, device='cuda'):
    global max_acc, max_model
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device, torch.float32)
            y_batch = y_batch.to(device).float().view(-1, 1)
              # BCE expects float [B, 1]

            optimizer.zero_grad()
            logits = model(x_batch)  # [B, 1]
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Evaluate on train and val
        #train_acc, train_f1 = evaluate(model, train_loader, device)
        val_acc, val_f1 = mi_evaluate(model, val_loader, device)
        if val_acc > max_acc:
            max_acc = val_acc
            torch.save(model.state_dict(), "best_model_mi.pt")

        print(
            f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f} | "
            #f"Train Acc: {train_acc:.2f}% - F1: {train_f1:.2f} | "
            f"Val Acc: {val_acc:.2f}% - F1: {val_f1:.2f}"
        )


train(model,train_loader,val_loader,100)
