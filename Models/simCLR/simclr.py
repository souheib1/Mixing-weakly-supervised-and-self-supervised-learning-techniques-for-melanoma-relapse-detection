# Implemented by Imen Mahdi
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from pytorch_metric_learning import losses





class SimCLR(nn.Module):
    def __init__(self, encoder, head, nb_channels, nb_rows, nb_columns):
        self.encoder = encoder
        self.head = head
        self.nb_channels = nb_channels
        self.nb_rows = nb_rows
        self.nb_columns = nb_columns

    def augmentation(self, X, p_flip=0.5, p_distort=0.5, p_blur=0.5):
        crop_size = self.nb_rows//2
        kernel_size=self.nb_rows//10

        transform = transforms.Compose([
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(p=p_flip),
            transforms.RandomVerticalFlip(p=p_flip),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.GaussianBlur(kernel_size)
        ])
        
        augmented_images = transform(X)
        
        return augmented_images

    def forward(self, x):
        view = self.augmentation(x)

        view_vectorized = view.view(-1, self.nb_channels * self.nb_rows * self.nb_columns)
        h = self.encoder(view_vectorized)

        z = self.head(h)
        return z
    

class Train_loader:
    def __init__(self):
        pass

    
def main():
    embeddings_dim = 10
    num_epochs = 100

    encoder = models.resnet50(pretrained=True)
    num_features = encoder.fc.in_features
    encoder.fc = nn.Linear(num_features, embeddings_dim)

    head = nn.Sequential(
            nn.Linear(embeddings_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embeddings_dim)
        )
    model = SimCLR(encoder, head)

    criterion = losses.NTXentLoss(temperature=0.07)
    optimizer = optim.LARS(model.parameters())
    train_loader = Train_loader()


    for epoch in range(num_epochs):
        running_loss = 0.0
        
        # Iterate over the training dataset
        for image in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            z1 = model(image)
            z2 = model(image)
            loss = criterion(z1, z2)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            epoch_loss = running_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

if __name__ == "__main__":
    main()