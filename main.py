import click
import torch
from torch import nn, optim
import matplotlib.pyplot as plt
from model import MyAwesomeModel
from torch.utils.data import DataLoader

from data import mnist


@click.group()
def cli():
    """Command line interface."""
    pass


@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
def train(lr):
    """Train a model on MNIST."""
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    model = MyAwesomeModel()
    train_set, _ = mnist()

    # generate the data loader
    train = DataLoader(train_set, batch_size=64, shuffle=True)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    epochs = 30
    steps = 0

    train_losses, test_losses = [], []
    print(torch.backends.mps.is_available())  # the MacOS is higher than 12.3+
    print(torch.backends.mps.is_built())

    model.train()  # Set the model to training mode
    for e in range(epochs):
        running_loss = 0
        print("Model:", model)
        for images, labels in train:
            optimizer.zero_grad()
            # print("images:", images)
            # print("images.shape:", images.shape)
            log_ps = model(images)
            print("log_ps:", log_ps)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        else:
            # at the end of each epoch the training loss is calculated
            # model.eval()  # Set the model to evaluation mode

            train_losses.append(running_loss / len(train_set))

            print(f"Epoch {e+1}/{epochs}.. " f"Train loss: {running_loss/len(train_set):.3f}.. ")

    torch.save(model.state_dict(), "final_checkpoint.pth")

    # After training, plot the training curve
    plt.plot(train_losses)
    plt.xlabel("Training Step")
    plt.ylabel("Training Loss")
    plt.show()

    return


@click.command()
@click.argument("model_checkpoint")
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

    test_losses.append(test_loss / len(testloader))

    accuracy = accuracy / len(testloader)

    print(f"Test loss: {test_loss/len(testloader):.3f}.. " f"Test accuracy: {accuracy:.3f}")
    print(f"Accuracy: {accuracy.item()*100}%")

    return


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
