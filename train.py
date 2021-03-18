import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
import numpy as np
from CamVidDataset import CamVidDataset
from models.SegNet import SegNet

device = torch.device('cuda:0')

def train(model, train_dl, test_dl, opt, loss_func, epochs):
    """ train model using using provided datasets, optimizer and loss function """
    train_loss = [0 for i in range(epochs)]
    test_loss  = [0 for i in range(epochs)]
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            loss = loss_func(model(xb), yb)
            print(loss)
            train_loss[epoch] = loss.item()
            loss.backward()
            opt.step()
            opt.zero_grad()
        with torch.no_grad():
            losses, nums = zip(*[(loss_func(model(xb.to(device)),yb.to(device)).item(),len(xb.to(device))) for xb, yb in test_dl])
            test_loss[epoch] = np.sum(np.multiply(losses, nums)) / np.sum(nums)
            correct = 0
            total = 0
            for data in test_dl:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'Epoch: {epoch+1}/{epochs}, Train Loss: {train_loss[epoch]}, Test Loss {test_loss[epoch]}, Accuracy: {100*correct/(total*768*1024)}')
    return train_loss, test_loss


if __name__ == "__main__":
    # Define Hyperparameters
    lr = 0.001
    bs = 2 
    epochs = 10
    # Load Data
    train_data = CamVidDataset(image_path='./CamVid/train', label_path='./CamVid/train_labels',transform=transforms.ToTensor())
    test_data = CamVidDataset(image_path='./CamVid/val', label_path='./CamVid/val_labels',transform=transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=bs, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=bs, shuffle=True)
    # Instantiate Model
    model = SegNet(32).to(device)
    #Define Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Train
    train(model, trainloader, testloader, optimizer, criterion,epochs)
    # Save Model to file
    with open('segnet.pkl', 'wb') as f:
        pickle.dump(model,f)
        f.close()
