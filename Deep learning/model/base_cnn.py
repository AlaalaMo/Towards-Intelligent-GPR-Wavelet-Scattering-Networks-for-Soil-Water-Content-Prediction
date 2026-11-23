import torch
import torch.nn as nn


class Base_CNN(nn.Module):
    def __init__(self, in_channels=1, input_size=8192, latent_dim=4, dropout_prob=0.4, print_shape=False):
        """
        Simplified CNN Encoder for reducing a 1D input vector to a latent representation.

        Args:
            in_channels (int): Number of input channels. Defaults to 1 for 1D input.
            input_size (int): Size of the input vector (e.g., 8192).
            latent_dim (int): Size of the latent output vector.
            dropout_prob (float): Dropout probability to prevent overfitting.
            print_shape (bool): Whether to print the intermediate tensor shapes during the forward pass.
        """
        super(Base_CNN, self).__init__()

        self.print_shape = print_shape

        self.local_conv1d = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=64, stride=64, padding=16),  # Local kernel
            nn.BatchNorm1d(32),
            nn.Tanh(),
            nn.Dropout(dropout_prob)
        )

        self.global_conv1d = nn.Sequential(
            nn.Conv1d(in_channels,32, kernel_size=256, stride=64, padding=127),  # Semi-global kernel
            nn.BatchNorm1d(32),
            nn.Tanh(),
            nn.Dropout(dropout_prob)
        )

        self.combine_features = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=8, stride=4, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout_prob)
        )

        # Reduce to latent space
        reduced_size = 3968 # 3 stride-2 reductions
        self.fc = nn.Sequential(
            nn.Linear(reduced_size, latent_dim),
            nn.ReLU()  # Output is the latent vector
        )

    def forward(self, x):
        """
        Forward pass through the encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Latent vector of shape (batch_size, latent_dim).
        """
        # Reshape input for 1D convolutions (batch_size, channels, sequence_length)
        x = x.view(x.size(0), 1, -1)

        # Local and semi-global feature extraction
        local_features = self.local_conv1d(x)
        global_features = self.global_conv1d(x)

        if self.print_shape:
            print("Local features:", local_features.shape)
            print("Global features:", global_features.shape)

        # Combine local and global features
        combined = torch.cat([local_features, global_features], dim=1)  # Concatenate along channel dimension
        combined = self.combine_features(combined)

        if self.print_shape:
            print("Combined features:", combined.shape)

        # Flatten and project to latent space
        combined = combined.view(combined.size(0), -1)
        latent = self.fc(combined)

        if self.print_shape:
            print("Latent vector:", latent.shape)

        return latent


class Stand_CNN(nn.Module):
    def __init__(self, in_channels=1, input_size=8192, latent_dim=4, dropout_prob=0.4, print_shape=False):
        """
        Standard CNN Encoder for reducing a 1D input vector to a latent representation.

        Args:
            in_channels (int): Number of input channels. Defaults to 1 for 1D input.
            input_size (int): Size of the input vector (e.g., 8192).
            latent_dim (int): Size of the latent output vector.
            dropout_prob (float): Dropout probability to prevent overfitting.
            print_shape (bool): Whether to print the intermediate tensor shapes during the forward pass.
        """
        super(Stand_CNN, self).__init__()

        self.print_shape = print_shape

        # Sequential convolutional layers with progressive reduction of spatial dimensions
        self.conv_blocks = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=8, stride=4, padding=2),  # First conv layer
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # Reduce dimensionality
            nn.Dropout(dropout_prob),

            nn.Conv1d(16, 32, kernel_size=8, stride=4, padding=2),  # Second conv layer
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout_prob),

            nn.Conv1d(32, 64, kernel_size=8, stride=4, padding=2),  # Third conv layer
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )

        # Flatten and reduce to latent space
        reduced_size = self._calculate_flattened_size(input_size, self.conv_blocks)
        self.fc = nn.Sequential(
            nn.Linear(reduced_size, latent_dim),
            nn.ReLU()  # Output latent vector
        )

    def _calculate_flattened_size(self, input_size, conv_blocks):
        """
        Calculate the flattened size after passing through convolutional layers.
        """
        dummy_input = torch.zeros(1, 1, input_size)  # (batch_size, channels, sequence_length)
        dummy_output = conv_blocks(dummy_input)
        # print(dummy_output.view(1,-1).size(1))
        return dummy_output.view(1, -1).size(1)

    def forward(self, x):
        """
        Forward pass through the encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Latent vector of shape (batch_size, latent_dim).
        """
        # Reshape input for 1D convolutions
        x = x.view(x.size(0), 1, -1)

        # Pass through convolutional layers
        x = self.conv_blocks(x)
        if self.print_shape:
            print("Shape after conv_blocks:", x.shape)

        # Flatten and project to latent space
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        if self.print_shape:
            print("Latent vector shape:", x.shape)

        return x