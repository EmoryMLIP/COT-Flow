import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gaussian_kernel(X: torch.Tensor, Y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    # Compute squared norms of each row in X and Y
    X_norm = (torch.norm(X, dim=1)**2).view(-1, 1)  # Shape: (n_samples_x, 1)
    Y_norm = (torch.norm(Y, dim=1)**2).view(1, -1)  # Shape: (1, n_samples_y)
    # Compute the pairwise squared Euclidean distances
    # dist_sq[i, j] = ||X_i||^2 + ||Y_j||^2 - 2 * X_i . Y_j
    dist_sq = X_norm + Y_norm - 2.0 * (X @ Y.t())
    # Compute the Gaussian kernel matrix
    K = torch.exp(-dist_sq / (2.0 * sigma**2))
    return K


def mmd(X: torch.Tensor, Y: torch.Tensor, kernel: str = "gaussian", sigma: float = 1.0) -> torch.Tensor:
    """
    Computes the Maximum Mean Discrepancy (MMD) between two empirical distributions.

    Args:
        X (torch.Tensor): A tensor of shape (n_samples_x, d), samples from the first distribution.
        Y (torch.Tensor): A tensor of shape (n_samples_y, d), samples from the second distribution.
        kernel (str): The type of kernel to use. Currently supports only "gaussian".
        sigma (float): The standard deviation for the Gaussian kernel.

    Returns:
        torch.Tensor: The MMD distance between the two distributions.
    """
    # Compute the pairwise kernel matrices
    if kernel == "gaussian":
        if sigma is None:
            K_X = gaussian_kernel(X, X)  # Shape (n_samples_x, n_samples_x)
            K_Y = gaussian_kernel(Y, Y)  # Shape (n_samples_y, n_samples_y)
            K_XY = gaussian_kernel(X, Y) # Shape (n_samples_x, n_samples_y)
        else:
            K_X = gaussian_kernel(X, X, sigma)  # Shape (n_samples_x, n_samples_x)
            K_Y = gaussian_kernel(Y, Y, sigma)  # Shape (n_samples_y, n_samples_y)
            K_XY = gaussian_kernel(X, Y, sigma) # Shape (n_samples_x, n_samples_y)
    # Compute the MMD^2 using the biased estimator
    mmd_squared = torch.mean(K_X) + torch.mean(K_Y) - 2.0 * torch.mean(K_XY)
    # Return the square root of the MMD^2 to get the MMD distance
    return torch.sqrt(mmd_squared)


if __name__ == "__main__":
    # Example usage:
    # Create some random data for demonstration
    torch.manual_seed(0)
    X = torch.randn(100, 5)  # 100 samples, 5-dimensional
    Y = torch.randn(100, 5)  # Another 100 samples, 5-dimensional
    mmd_value = mmd(X, Y, kernel="gaussian", sigma=1.0)
    print("MMD distance between X and Y:", mmd_value.item())
