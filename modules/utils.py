import numpy as np
import cv2
import torch


def penalty_builder(penalty_config):
    if penalty_config == '':
        return lambda x, y: y
    pen_type, alpha = penalty_config.split('_')
    alpha = float(alpha)
    if pen_type == 'wu':
        return lambda x, y: length_wu(x, y, alpha)
    if pen_type == 'avg':
        return lambda x, y: length_average(x, y, alpha)


def length_wu(length, logprobs, alpha=0.):
    """
    NMT length re-ranking score from
    "Google's Neural Machine Translation System" :cite:`wu2016google`.
    """

    modifier = (((5 + length) ** alpha) /
                ((5 + 1) ** alpha))
    return logprobs / modifier


def length_average(length, logprobs, alpha=0.):
    """
    Returns the average probability of tokens in a sequence.
    """
    return logprobs / length


def split_tensors(n, x):
    if torch.is_tensor(x):
        assert x.shape[0] % n == 0
        x = x.reshape(x.shape[0] // n, n, *x.shape[1:]).unbind(1)
    elif type(x) is list or type(x) is tuple:
        x = [split_tensors(n, _) for _ in x]
    elif x is None:
        x = [None] * n
    return x


def repeat_tensors(n, x):
    """
    For a tensor of size Bx..., we repeat it n times, and make it Bnx...
    For collections, do nested repeat
    """
    if torch.is_tensor(x):
        x = x.unsqueeze(1)  # Bx1x...
        x = x.expand(-1, n, *([-1] * len(x.shape[2:])))  # Bxnx...
        x = x.reshape(x.shape[0] * n, *x.shape[2:])  # Bnx...
    elif type(x) is list or type(x) is tuple:
        x = [repeat_tensors(n, _) for _ in x]
    return x


def generate_heatmap(image, weights):
    # print(image.shape)
    # print(weights.shape)
    image = image.transpose(1, 2, 0)
    height, width, _ = image.shape
    weights = weights.reshape(int(weights.shape[0] ** 0.5), int(weights.shape[0] ** 0.5))
    weights = weights - np.min(weights)
    # print(weights.shape)
    # print(np.max(weights))
    weights = weights / np.nanmax(weights)
    weights = cv2.resize(weights, (width, height))
    weights = np.uint8(255 * weights)
    heatmap = cv2.applyColorMap(weights, cv2.COLORMAP_JET)
    result = heatmap * 0.5 + image * 0.5
    return result

# # def generate_heatmap(image, weights):
# #     # print(image.shape)
# #     # print(weights.shape)
    
# #     # Check if weights shape is compatible with reshaping
# #     side_length = int(np.sqrt(weights.shape[0]))
# #     if side_length * side_length != weights.shape[0]:
# #         raise ValueError("Invalid weights shape. Expected a perfect square number of weights.")

# #     # Transpose image and get dimensions
# #     image = image.transpose(1, 2, 0)
# #     height, width, _ = image.shape
    
# #     # Reshape and normalize weights
# #     weights = weights.reshape(side_length, side_length)
# #     weights = weights - np.min(weights)
# #     max_weight = np.max(weights)
    
# #     # Avoid division by zero
# #     if max_weight != 0:
# #         weights = weights / max_weight
# #     else:
# #         weights = np.zeros_like(weights)  # Set all weights to zero
        
# #     # Resize and convert weights to uint8
# #     weights = cv2.resize(weights, (width, height))
# #     weights = np.uint8(255 * weights)
    
# #     # Create heatmap and blend with image
# #     heatmap = cv2.applyColorMap(weights, cv2.COLORMAP_JET)
    
# #     # Convert heatmap to the same data type as image
# #     heatmap = heatmap.astype(image.dtype)
    
# #     # Blend heatmap with image
# #     result = cv2.addWeighted(heatmap, 0.5, image, 0.5, 0)
    
# #     return result
# def generate_heatmap(image, weights):
#     # Calculate the size of the heatmap grid (a perfect square)
#     num_weights = len(weights)
#     side_length = int(np.sqrt(num_weights))
    
#     # Reshape the weights to match the grid size
#     reshaped_weights = np.reshape(weights, (side_length, side_length))
    
#     # Resize the image to match the heatmap size
#     heatmap_size = reshaped_weights.shape[0]
#     resized_image = cv2.resize(image, (heatmap_size, heatmap_size))
    
#     # Normalize the weights to [0, 255]
#     normalized_weights = (reshaped_weights - reshaped_weights.min()) / (reshaped_weights.max() - reshaped_weights.min())
#     heatmap = np.uint8(normalized_weights * 255)
    
#     # Apply colormap to the heatmap
#     heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
#     # Overlay heatmap on the resized image
#     heatmap_with_overlay = cv2.addWeighted(resized_image, 0.7, heatmap_colored, 0.3, 0)
    
#     return result

