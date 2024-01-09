import torch
from ml_ops_cookie.models.model import MyNeuralNet
from torch.utils.data import DataLoader
from torch import nn
import click

@click.group()
def cli():
    """Command line interface."""
    pass


@click.command()
@click.option("--model_checkpoint", default="models/20240109_105514/final_checkpoint.pth", help="learning rate to use for training")
@click.option("--dataloader", default="data/processed/test_dataset.pt", help="learning rate to use for training")
def predict(
    model_checkpoint: str,
    dataloader: str
) -> None:
    """Run prediction for a given model and dataloader.
    
    Args:
        model: model to use for prediction
        dataloader: dataloader with batches
    
    Returns
        Tensor of shape [N, d] where N is the number of samples and d is the output dimension of the model

    """

    model = MyNeuralNet()
    state_dict = torch.load(model_checkpoint)
    print(state_dict.keys())
    model.load_state_dict(state_dict)

    test_dataset = torch.load(f"{dataloader}")
    testloader = DataLoader(test_dataset, batch_size=64, shuffle=True)
    test_loss = 0
    test_losses = []
    accuracy = 0
    criterion = nn.NLLLoss()

    model.eval()
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

    # return torch.cat([model(batch) for batch in dataloader], 0)
    return


cli.add_command(predict)

if __name__ == "__main__":
    cli()