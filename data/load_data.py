from .load_dataset import load_dataset
from torch.utils.data import DataLoader

def load_data(data_config, batch_size, pin_memory = True):

    train, val = load_dataset(data_config)

    train = DataLoader(
                        train,
                        batch_size = batch_size,
                        shuffle = True,
                        num_workers = 2,
                        pin_memory = pin_memory,
                      )

    val = DataLoader(
                        val,
                        batch_size = 1,
                        shuffle = False,
                        num_workers = 2,
                        pin_memory = pin_memory,
                    )
        
    return train, val