from torch.utils.data import DataLoader



class VGGDataLoader(DataLoader):
    def __int__(self, size, dataset, batch_size, shuffle):
        self.size = size
        super(VGGDataLoader).__init__(dataset, batch_size, shuffle)


train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)