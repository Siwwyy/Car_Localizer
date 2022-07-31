import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch

#Initialize function to calculate the mean and std - used in normalization process:
def get_mean_and_std(dataloader):
    """Returns a mean and std of train dataset in order to perform normalization on final train dataset.
            Parameters
            ----------
            dataloader : torch tensor collections
                train_raw dataset transformed to torch tensor and prev-resized
            Returns
            -------
                Mean and Std in  list format
    """
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in dataloader:
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1
    # mean: the following function
    mean = channels_sum / num_batches
    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
    mean = torch.Tensor(mean).tolist()
    std = torch.Tensor(std).tolist()
    return mean, std

#Calculation ACC function
def binary_acc(y_pred, y_test):
    """Returns a accuracy metric ( pl. dokladnosc modelu) on specified gpu_idx.
        Parameters
        ----------
        y_pred : torch tensor
            torch tensor predicted values
        y_test: torch tensor
            torch tensor from original test dataset (translated on torch.tensor()
        Returns
        -------
            Specific ACC value (binary clasisification)
    """

    y_pred_tag = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_tag, dim = 1)
    correct_results_sum = (y_pred_tags == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    return acc