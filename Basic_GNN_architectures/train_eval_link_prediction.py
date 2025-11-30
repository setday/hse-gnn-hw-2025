import torch
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score


def evaluate_link_prediction(model, decoder, data):
    """
    Оценка link prediction модели (dot-product decoder) по AUC и Average Precision.
    """
    model.eval()
    decoder.eval()
    with torch.no_grad():
        z = model(data.x, data.edge_index)

        # Пары узлов
        pos_edge = data.pos_edge_label_index
        neg_edge = data.neg_edge_label_index

        # Скалярное произведение (dot product)
        # Это декодер, который предсказывает связь между узлами:
        # если узлы похожи, их эмбеддинги близки → dot product большой
        # если узлы несвязаны, dot product маленький
        # Это стандартный метод в GAE/VGAE, Node2Vec, DeepWalk, LINE и многих GNN-подходах.
        # pos_score = (z[pos_edge[0]] * z[pos_edge[1]]).sum(dim=1)
        # neg_score = (z[neg_edge[0]] * z[neg_edge[1]]).sum(dim=1)
        # scores = torch.cat([pos_score, neg_score], dim=0)

        # Формируем правильные метки:
        # 1 для позитивных рёбер
        # 0 для негативных рёбер
        # labels = torch.cat([
        #     torch.ones(pos_score.size(0)),
        #     torch.zeros(neg_score.size(0))
        # ]).to(scores.device)
        labels, scores = decoder(z, pos_edge, neg_edge)

        # Превращаем скалярные произведения в вероятности
        probs = torch.sigmoid(scores)
        auc = roc_auc_score(labels.cpu(), probs.cpu())
        ap = average_precision_score(labels.cpu(), probs.cpu())

        return auc, ap

import copy

def train_link_prediction(model, decoder, train_data, val_data, optimizer, epochs=100, patience=10):
    """
    Обучение модели link prediction с:
      - binary cross entropy loss
      - валидацией по AUC и BCE
      - ранней остановкой
      - сохранением лучших весов модели

    Args:
        model: GNN энкодер (например, GCNEncoder)
        train_data: Data объект для обучения
        val_data: Data объект для валидации
        optimizer: torch.optim.Adam / SGD
        epochs: максимальное число эпох
        patience: сколько эпох ждать улучшения val_loss или val_auc

    Returns:
        train_losses, val_losses, val_aucs
    """

    train_losses = []
    val_losses = []
    val_aucs = []

    best_val_auc = 0.0
    best_model_state = None
    epochs_no_improve = 0

    for epoch in range(epochs):

        # TRAIN
        model.train()
        decoder.train()
        optimizer.zero_grad()

        # Эмбеддинги train-графа
        z = model(train_data.x, train_data.edge_index)

        # Позитивные/негативные пары
        pos_edge = train_data.pos_edge_label_index
        neg_edge = train_data.neg_edge_label_index

        # Dot-product decoder
        # pos_score = (z[pos_edge[0]] * z[pos_edge[1]]).sum(dim=1)
        # neg_score = (z[neg_edge[0]] * z[neg_edge[1]]).sum(dim=1)
        # scores = torch.cat([pos_score, neg_score], dim=0)

        # labels = torch.cat([
        #     torch.ones(pos_score.size(0)),
        #     torch.zeros(neg_score.size(0))
        # ]).to(scores.device)
        labels, scores = decoder(z, pos_edge, neg_edge)

        # BCE loss
        loss = F.binary_cross_entropy_with_logits(scores, labels)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        # VALIDATION
        model.eval()
        decoder.eval()
        with torch.no_grad():

            # AUC/AP — более правильная метрика для link prediction
            val_auc, val_ap = evaluate_link_prediction(model, decoder, val_data)
            val_aucs.append(val_auc)

            # А также считаем валидизационный BCE loss (для более стабильной ранней остановки)
            z_val = model(val_data.x, val_data.edge_index)
            pos_val = val_data.pos_edge_label_index
            neg_val = val_data.neg_edge_label_index

            pos_val_score = (z_val[pos_val[0]] * z_val[pos_val[1]]).sum(dim=1)
            neg_val_score = (z_val[neg_val[0]] * z_val[1]).sum(dim=1)

            val_scores = torch.cat([pos_val_score, neg_val_score], dim=0)
            val_labels = torch.cat([
                torch.ones(pos_val_score.size(0)),
                torch.zeros(neg_val_score.size(0))
            ]).to(scores.device)

            val_loss = F.binary_cross_entropy_with_logits(val_scores, val_labels)
            val_losses.append(val_loss.item())


        print(f"Epoch {epoch:03d}, "
              f"Train Loss: {loss.item():.4f}, "
              f"Val Loss: {val_loss.item():.4f}, "
              f"Val AUC: {val_auc:.4f}")


        # Вариант: используем метрику AUC как главный критерий
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Если долго нет улучшений — остановить обучение
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch} (best AUC = {best_val_auc:.4f})")
            break

    # LOAD BEST MODEL
    if best_model_state is not None:
        print("Loading the best model weights...")
        model.load_state_dict(best_model_state)

    return train_losses, val_losses, val_aucs

