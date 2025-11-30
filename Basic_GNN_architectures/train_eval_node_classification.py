import copy

import torch
import torch.nn.functional as F

def train_node_classification(model, data, optimizer, epochs=200, patience=20):
    best_val_acc = 0.0
    best_model_state = None
    epochs_no_improve = 0

    train_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        # TRAIN
        model.train()
        optimizer.zero_grad()

        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        # VALIDATION
        model.eval()
        val_acc = evaluate_node_classification(model, data, data.val_mask)
        val_accuracies.append(val_acc)


        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d}, Loss: {loss.item():.4f}, Val Acc: {val_acc:.4f}")


        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break


    if best_model_state:
        model.load_state_dict(best_model_state)

    return train_losses, val_accuracies



def evaluate_node_classification(model, data, mask):
    """
    Оценка точности модели на подмножестве узлов, заданном маской.

    Args:
        model: обученная GNN-модель
        data: граф
        mask: булевый тензор маски (train_mask, val_mask, test_mask)

    Returns:
        float: accuracy на выбранной части графа
    """
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out[mask].argmax(dim=1)
        acc = (pred == data.y[mask]).sum().item() / mask.sum().item()
        return acc
