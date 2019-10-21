import torch

class MultiTransDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform_big, transform_small):
        self.data = dataset
        self.transform_big = transform_big
        self.transform_small = transform_small

    def __getitem__(self, index):
        image, target = self.data[index]
        big_image = self.transform_big(image)
        small_image = self.transform_small(image)
        return big_image, small_image, target

    def __len__(self):
        return(len(self.data))

def collate_fn(data):
    big_image_list, small_image_list, target_list = list(zip(*data))
    big_images = torch.stack(big_image_list)
    small_images = torch.stack(small_image_list)
    targets = torch.stack(target_list)
    return big_images, small_images, targets

