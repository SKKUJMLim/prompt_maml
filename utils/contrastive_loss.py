import torch
import torch.nn.functional as F


def soft_nearest_neighbors_loss_cos_similarity(features, labels, temperature=0.1):
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

    # Compute pairwise similarity (cosine similarity)
    similarity = torch.mm(features, features.t())

    # Scale similarity by temperature
    scaled_similarity = similarity / temperature

    # Create a mask to exclude self-similarity
    batch_size = features.size(0)
    mask = torch.eye(batch_size, device=features.device).bool()

    # Convert labels to one-hot encoding
    labels = labels.unsqueeze(1)
    one_hot_labels = (labels == labels.t()).float()

    # Apply the mask to exclude self-similarity
    exp_similarity = torch.exp(scaled_similarity) * ~mask

    # Compute the denominators for the softmax
    denominators = exp_similarity.sum(dim=1, keepdim=True)

    # Compute the soft nearest neighbors loss
    positive_similarity = exp_similarity * one_hot_labels
    positive_probabilities = positive_similarity.sum(dim=1) / denominators.squeeze(1)

    # Take the log and compute the mean loss
    loss = -torch.log(positive_probabilities + 1e-8).mean()

    return loss

def soft_nearest_neighbors_loss_euclidean(features, labels, temperature=0.1):
    """
    Compute the Soft Nearest Neighbors Loss using Euclidean distance.

    Args:
        features (torch.Tensor): Feature vectors with shape (batch_size, feature_dim).
        labels (torch.Tensor): Labels for the features with shape (batch_size,).
        temperature (float): Temperature scaling factor for the softmax.

    Returns:
        torch.Tensor: The computed loss.
    """
    # Compute pairwise Euclidean distance manually
    batch_size = features.size(0)
    distance_matrix = torch.norm(features.unsqueeze(1) - features.unsqueeze(0), dim=2, p=2)

    # Scale distance by temperature (inverse to make closer distances more influential)
    scaled_distances = -distance_matrix / temperature

    # Create a mask to exclude self-similarity
    mask = torch.eye(batch_size, device=features.device).bool()

    # Convert labels to one-hot encoding
    labels = labels.unsqueeze(1)
    one_hot_labels = (labels == labels.t()).float()

    # Apply the mask to exclude self-similarity
    exp_scaled_distances = torch.exp(scaled_distances) * ~mask

    # Compute the denominators for the softmax
    denominators = exp_scaled_distances.sum(dim=1, keepdim=True)

    # Compute the soft nearest neighbors loss
    positive_distances = exp_scaled_distances * one_hot_labels
    positive_probabilities = positive_distances.sum(dim=1) / denominators.squeeze(1)

    # Take the log and compute the mean loss
    loss = -torch.log(positive_probabilities + 1e-8).mean()

    return loss

