import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import numpy as np


class BuildingAutoencoder(nn.Module):
    """
    Convolutional Autoencoder for building footprint representation.
    """
    def __init__(self, img_size=64, latent_dim=64):
        super(BuildingAutoencoder, self).__init__()
        
        self.img_size = img_size
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Calculate flattened feature dimensions
        self.flatten_dim = 128 * (img_size // 8) * (img_size // 8)
        
        # Latent space
        self.fc1 = nn.Linear(self.flatten_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, self.flatten_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
    
    def decode(self, x):
        x = self.fc2(x)
        x = x.view(x.size(0), 128, self.img_size // 8, self.img_size // 8)
        x = self.decoder(x)
        return x
    
    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded


def train_autoencoder(model, train_loader, device, num_epochs=50):
    """
    Train the autoencoder model.
    
    Args:
        model (BuildingAutoencoder): Autoencoder model
        train_loader (DataLoader): Training data loader
        device (torch.device): Device to use for training
        num_epochs (int): Number of training epochs
        
    Returns:
        model: Trained model
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for data in train_loader:
            img = data[0].to(device)
            
            # Forward pass
            output = model(img)
            loss = criterion(output, img)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.6f}')
    
    return model


def extract_autoencoder_features(model, image_tensors, device):
    """
    Extract feature vectors using the trained autoencoder.
    
    Args:
        model (BuildingAutoencoder): Trained autoencoder model
        image_tensors (torch.Tensor): Image tensor data
        device (torch.device): Device to use for processing
        
    Returns:
        np.ndarray: Feature vectors
    """
    model.eval()
    with torch.no_grad():
        tensors = image_tensors.to(device)
        vectors = model.encode(tensors).cpu().numpy()
    
    return vectors


class ResNetFeatureExtractor(nn.Module):
    """
    ResNet-based feature extractor for building footprints.
    """
    def __init__(self, output_dim=128):
        super(ResNetFeatureExtractor, self).__init__()
        # Load pretrained ResNet18
        resnet = models.resnet18(pretrained=True)
        # Remove the last fully connected layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        # Add new fully connected layer to reduce feature dimension to specified size
        self.fc = nn.Linear(512, output_dim)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def extract_resnet_features(model, images, preprocess, device):
    """
    Extract feature vectors using the ResNet feature extractor.
    
    Args:
        model (ResNetFeatureExtractor): ResNet feature extractor model
        images (list): List of images
        preprocess (transforms.Compose): Preprocessing transformations
        device (torch.device): Device to use for processing
        
    Returns:
        np.ndarray: Feature vectors
    """
    model.eval()  # Set to evaluation mode
    vectors = []
    
    with torch.no_grad():
        for img in images:
            # Preprocess image
            img_tensor = preprocess(img).unsqueeze(0).to(device)
            # Extract features
            features = model(img_tensor).cpu().numpy()[0]
            vectors.append(features)
    
    return np.array(vectors)
