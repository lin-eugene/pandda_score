from lib.extract_box import create_numpy_array_with_gemmi_interpolate
from learning.torch_data_setup import *
from torch.utils.data import DataLoader

def get_random_sample(dataframe):
    training_dataset = DebugResidueDataset(residues_dframe=dataframe, 
                                    transform=transforms.Compose([
                                        SamplingRandomRotations(),
                                        ConcatEventResidueToTwoChannels(),
                                        ToTensor()
                                    ])) # -> event_residue_array has 4 dims [C, D, W, H] (Channels, C = 2; D = 1; W = 32; H = 32)


    # sample = training_dataset[1]
    train_dataloader = DataLoader(dataset=training_dataset, 
                        batch_size=1, 
                        num_workers=1,
                        shuffle=True) # -> event_residue_array has 5 dims [N, C, D, W, H] (N_batches, N = 1; C = 2; D = 1; W = 32; H = 32)

    sample = next(iter(train_dataloader))

    return sample