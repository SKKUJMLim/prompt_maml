import torch
import torch.nn.functional as F


# def soft_nearest_neighbors_loss_cos_similarity(features, labels, temperature):
#     """
#     Compute the Soft Nearest Neighbors Loss.
#
#     Args:
#         features (torch.Tensor): Feature vectors with shape (batch_size, feature_dim).
#         labels (torch.Tensor): Labels for the features with shape (batch_size,).
#         temperature (float): Temperature scaling factor for the softmax.
#
#     Returns:
#         torch.Tensor: The computed loss.
#     """
#     # Normalize features to ensure unit vectors
#     features = F.normalize(features, dim=1)
#
#     # Compute pairwise similarity (cosine similarity)
#     similarity = torch.mm(features, features.t())
#
#     # Scale similarity by temperature
#     scaled_similarity = similarity / temperature
#
#     # Create a mask to exclude self-similarity
#     batch_size = features.size(0)
#     mask = torch.eye(batch_size, device=features.device).bool()
#
#     # Convert labels to one-hot encoding
#     labels = labels.unsqueeze(1)
#     one_hot_labels = (labels == labels.t()).float()
#
#     # Apply the mask to exclude self-similarity
#     exp_similarity = torch.exp(scaled_similarity) * ~mask
#
#     # Compute the denominators for the softmax
#     denominators = exp_similarity.sum(dim=1, keepdim=True)
#
#     # Compute the soft nearest neighbors loss
#     positive_similarity = exp_similarity * one_hot_labels
#     positive_probabilities = positive_similarity.sum(dim=1) / denominators.squeeze(1)
#
#     # Take the log and compute the mean loss
#     loss = -torch.log(positive_probabilities + 1e-8).mean()
#
#     return loss


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