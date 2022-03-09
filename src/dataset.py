from torch.utils.data import Dataset


class AfriT5Dataset(Dataset):
    def __init__(self, source) -> None:
        super(AfriT5Dataset, self).__init__()
        self.source = source

        with open(self.source) as file:
            text = file.read()

        self.text = text.split("\n")

    def __getitem__(self, index):
        return self.text[index]

    def __len__(self):
        return len(self.text)
