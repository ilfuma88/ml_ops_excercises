import click
import torch
from torch import nn, optim
import matplotlib.pyplot as plt
from ml_ops_cookie.models.model import MyNeuralNet
from torch.utils.data import DataLoader
import os
from datetime import datetime


def train(lr):
    """Train a model on MNIST."""
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    model = MyNeuralNet()
    # train_set, _ = mnist()
    train_set = torch.load('./data/processed/train_dataset.pt')

        # generate the data loader
    train = DataLoader(train_set, batch_size=64, shuffle=True)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    epochs = 17
    steps = 0

    train_losses, test_losses = [], []
    print(torch.backends.mps.is_available()) #the MacOS is higher than 12.3+
    print(torch.backends.mps.is_built())

    model.train()  # Set the model to training mode
    for e in range(epochs):
        running_loss = 0
        # print("Model:", model)
        for images, labels in train:
            
            optimizer.zero_grad()
            # print("images:", images)
            # print("images.shape:", images.shape)    
            log_ps = model(images)
            # print("log_ps:", log_ps)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        else:
            #at the end of each epoch the training loss is calculated
            # model.eval()  # Set the model to evaluation mode
            
            train_losses.append(running_loss/len(train_set))
            
            print(f"Epoch {e+1}/{epochs}.. "
                f"Train loss: {running_loss/len(train_set):.3f}.. "
                )
            
    # Get current date and time
    now = datetime.now()
    # Format as string
    date_time = now.strftime("%Y%m%d_%H%M%S")
    # Define directory path
    dir_path = f'models/{date_time}'
    # Define directory path fir image
    dir_path_image = f'reports/figures/{date_time}'
    # Create directory
    os.makedirs(dir_path, exist_ok=True)
    os.makedirs(dir_path_image, exist_ok=True)

    # Save the model
    torch.save(model.state_dict(), f'{dir_path}/final_checkpoint.pth')
    
    # After training, plot the training curve
    plt.plot(train_losses)
    plt.xlabel('Training Step')
    plt.ylabel('Training Loss')
    plt.savefig(f'{dir_path_image}/training_loss.png')
    plt.show()

    return




def evaluate(model_checkpoint):
    """Evaluate a trained model."""
    print("Evaluating like my life dependends on it")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    state_dict = torch.load(model_checkpoint)

    print(state_dict.keys())

    model = MyAwesomeModel()

    model.load_state_dict(state_dict)

    _, test_set = mnist()
    testloader = DataLoader(test_set, batch_size=64, shuffle=True)
    test_loss = 0
    test_losses = []
    accuracy = 0
    criterion = nn.NLLLoss()


    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for images, labels in testloader:  
            log_ps = model(images)
            test_loss += criterion(log_ps, labels)
            
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))

    test_losses.append(test_loss/len(testloader))

    accuracy = accuracy/len(testloader)
    
    print(
            f"Test loss: {test_loss/len(testloader):.3f}.. "
            f"Test accuracy: {accuracy:.3f}")
    print(f'Accuracy: {accuracy.item()*100}%')

    return



if __name__ == "__main__":
    train(1e-3)
