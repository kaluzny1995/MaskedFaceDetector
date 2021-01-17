import torch


class MaskedFacesExamples(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X = self.data[idx]

        if self.transform:
            X = self.transform(X)

        return X


class Config:
    def __init__(self):
        # CNN (cnn)
        self.cnn_in_channels = 3
        self.cnn_out_channels = 256
        self.cnn_kernel_size = 9

        # Primary Capsule (pc)
        self.pc_num_capsules = 8
        self.pc_in_channels = 256
        self.pc_out_channels = 32
        self.pc_kernel_size = 9
        self.pc_num_routes = 32 * 8 * 8

        # Digit Capsule (dc)
        self.dc_num_capsules = 2
        self.dc_num_routes = 32 * 8 * 8
        self.dc_in_channels = 8
        self.dc_out_channels = 16

        # Decoder
        self.input_width = 32
        self.input_height = 32

