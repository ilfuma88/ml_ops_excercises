import torch
import glob
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset


def mnist():
    """Return train and test dataloaders for MNIST."""
    # exchange with the corrupted mnist dataset
    # train = torch.randn(50000, 784)
    # test = torch.randn(10000, 784)

    data_folder = "/Users/riccardoconti/Documents/DTU/MLOPS/dtu_mlops/data/corruptmnist"
    output_folder = "../MachineLearningOperations/cookiecutterproject/data//processed"

    train_images_pattern = "./data/corruptmnist/train_images_*.pt"
    train_target_pattern = "./data/corruptmnist/train_target_*.pt"
    test_images_pattern = "./data/corruptmnist/test_images.pt"
    test_target_pattern = "./data/corruptmnist/test_target.pt"

    # List all files matching the pattern
    train_images_files = glob.glob(train_images_pattern)
    train_target_files = glob.glob(train_target_pattern)

    test_images_files = glob.glob(test_images_pattern)
    test_target_files = glob.glob(test_target_pattern)

    # Load and concatenate all the data tensors
    print(train_images_files)
    # train = torch.cat([torch.load(f) for f in train_images_files])
    # train_target = torch.cat([torch.load(f) for f in train_target_files])
    train_images = torch.cat([torch.load(f"{data_folder}/train_images_{i}.pt") for i in range(6)], dim=0)
    train_target = torch.cat([torch.load(f"{data_folder}/train_target_{i}.pt") for i in range(6)], dim=0)

    # test = torch.cat([torch.load(f) for f in test_images_files])
    # test_target = torch.cat([torch.load(f) for f in test_target_files])

    test_images = torch.load(f"{data_folder}/test_images.pt")
    test_target = torch.load(f"{data_folder}/test_target.pt")

    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # print(train_images.shape)
    # print(train_images)
    # print(train_target.shape)
    # print(train_target)
    # zipping togeter the data and the target
    train = TensorDataset(train_images, train_target)
    test = TensorDataset(test_images, test_target)

    return train, test


# read the tensors from the .pt files in /raw
# cat all the tensors in one tensor, normalize them and save them in /processed
# split the tensor in train and test and save them in /processed
def process_data():
    data_folder = "./data/raw"
    output_folder = "./data//processed"

    train_images = torch.cat([torch.load(f"{data_folder}/train_images_{i}.pt") for i in range(6)], dim=0)
    train_target = torch.cat([torch.load(f"{data_folder}/train_target_{i}.pt") for i in range(6)], dim=0)

    test_images = torch.load(f"{data_folder}/test_images.pt")
    test_target = torch.load(f"{data_folder}/test_target.pt")

    # Assuming that train_images and test_images are tensors
    train_images = torch.nn.functional.normalize(train_images, p=2, dim=1)
    test_images = torch.nn.functional.normalize(test_images, p=2, dim=1)

    # print(train_images.shape)
    # print(train_images)
    # print(train_target.shape)
    # print(train_target)

    # print(test_images.shape)
    # print(test_images)
    # print(test_target.shape)
    # print(test_target)

    # zipping togeter the data and the target
    train = TensorDataset(train_images, train_target)
    test = TensorDataset(test_images, test_target)

    # print(train)
    # print(test)

    # Save the train and test datasets
    torch.save(train, f"{output_folder}/train_dataset.pt")
    torch.save(test, f"{output_folder}/test_dataset.pt")

    return train, test


if __name__ == "__main__":
    # Get the data and process it
    process_data()
