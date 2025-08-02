import torch
from sklearn.metrics import f1_score


def mi_evaluate(model, dataloader, device='cuda'):
    model.eval()
    correct = total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device, torch.float32)
            y_batch = y_batch.to(device).long().view(-1)  # Integer class labels

            logits = model(x_batch)              # Shape: [B, num_classes]
            preds = torch.argmax(logits, dim=1)  # Shape: [B]

            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    accuracy = 100.0 * correct / total
    f1 = f1_score(all_labels, all_preds, average='weighted')  # or 'macro'/'micro' if needed

    return accuracy, f1
