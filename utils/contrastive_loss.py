import torch
import torch.nn.functional as F


def cosine_similarity_classifier(features, class_prototypes):
    similarities = F.cosine_similarity(features.unsqueeze(1), class_prototypes.unsqueeze(0), dim=-1)
    return similarities.argmax(dim=1)  # 가장 유사한 클래스 반환


def compute_class_prototypes(preds, target, num_classes, device):


    class_features = {i: [] for i in range(num_classes)}

    for i, label in enumerate(target):
        class_features[label.item()].append(preds[i].cpu())

    # 각 클래스의 평균 벡터를 대표 벡터로 설정
    class_prototypes = torch.stack([torch.stack(class_features[i]).mean(dim=0) for i in range(num_classes)])

    return class_prototypes.to(device)


def soft_nearest_neighbors_loss_cos_similarity(features, labels, temperature):
    """
    Compute the Soft Nearest Neighbors Loss.

    Args:
        features (torch.Tensor): Feature vectors with shape (batch_size, feature_dim).
        labels (torch.Tensor): Labels for the features with shape (batch_size,).
        temperature (float): Temperature scaling factor for the softmax.

    Returns:
        torch.Tensor: The computed loss.
    """
    # Normalize features to ensure unit vectors
    features = F.normalize(features, dim=1)

    # Compute pairwise cosine similarity
    similarity = torch.mm(features, features.t())

    # Scale similarity by temperature
    scaled_similarity = similarity / temperature

    # Create a mask to exclude self-similarity
    batch_size = features.size(0)
    mask = torch.eye(batch_size, device=features.device).bool()

    # Convert labels to one-hot encoding (pairwise similarity mask)
    labels = labels.unsqueeze(1)
    one_hot_labels = (labels == labels.t()).float()

    # Apply the mask to exclude self-similarity
    exp_similarity = torch.exp(scaled_similarity)
    exp_similarity = exp_similarity.masked_fill(mask, 0)  # Exclude diagonal (self-similarity)

    # Compute the denominators for the softmax
    denominators = exp_similarity.sum(dim=1, keepdim=True)

    # Compute the numerator: positive pair similarities
    positive_similarity = exp_similarity * one_hot_labels
    numerators = positive_similarity.sum(dim=1)

    # Compute the probabilities
    positive_probabilities = numerators / (denominators.squeeze(1) + 1e-8)

    # Take the log and compute the mean loss
    loss = -torch.log(positive_probabilities + 1e-8).mean()

    return loss


def squared_euclidean_distance(x):
    """
    x: (N, D) 형태의 텐서
    반환값: (N, N) 형태의 유클리드 제곱 거리 행렬
    """
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * (a @ b.T)
    x_norm = (x ** 2).sum(dim=1, keepdim=True)  # ||x||^2, Shape: (N, 1)
    distance_matrix = x_norm + x_norm.T - 2 * (x @ x.T)  # (N, N)
    distance_matrix = torch.sqrt(torch.clamp(distance_matrix, min=1e-8))  # sqrt 적용 (음수 방지)
    return distance_matrix


def soft_nearest_neighbors_loss_euclidean(features, labels, temperature):
    """
    Computes the Soft Nearest Neighbors Loss using Euclidean distance.

    Args:
        features (torch.Tensor): Tensor of shape (N, D) where N is the number of samples and D is the embedding dimension.
        labels (torch.Tensor): Tensor of shape (N,) containing the class labels for the samples.
        temperature (float): Temperature parameter to control the sharpness of the softmax distribution.

    Returns:
        torch.Tensor: The computed loss value.
    """
    # Ensure inputs are on the same device
    device = features.device
    labels = labels.to(device)

    # Compute pairwise Euclidean distances
    # distances = torch.cdist(features, features, p=2) ** 2  # Shape: (N, N)
    distances = squared_euclidean_distance(features)

    # Apply softmax with temperature to the negative distances
    negative_distances = -distances / temperature
    similarity = F.softmax(negative_distances, dim=1)  # Shape: (N, N)

    # Create a mask for same-class pairs
    label_mask = labels.unsqueeze(0) == labels.unsqueeze(1)  # Shape: (N, N)

    # Remove diagonal elements from the mask
    label_mask = label_mask.fill_diagonal_(False)

    # Compute the loss
    numerator = torch.sum(similarity * label_mask.float(), dim=1)
    denominator = torch.sum(similarity, dim=1)

    # Avoid numerical issues by adding a small epsilon
    epsilon = 1e-8
    loss = -torch.log((numerator + epsilon) / (denominator + epsilon))

    return loss.mean()


def info_nce_loss(z, labels, temperature=0.07):
    z = F.normalize(z, dim=1)  # normalize
    sim_matrix = torch.matmul(z, z.T)  # similarity
    sim_matrix = sim_matrix / temperature

    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float()

    logits = sim_matrix
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(z.size(0)).view(-1, 1).to(z.device),
        0
    )
    mask = mask * logits_mask

    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-7)

    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-7)

    loss = -mean_log_prob_pos
    loss = loss.mean()
    return loss