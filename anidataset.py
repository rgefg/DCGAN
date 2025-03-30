from PIL import Image
import os
from torch.utils.data import Dataset
class anidataset(Dataset):
    def __init__(self,datapath,transform):
        self.file_list=list()     #路径列表
        files=os.listdir(datapath)
        for file in files:
            path=os.path.join(datapath,file)
            self.file_list.append(path)
        self.transform=transform
        self.length=len(self.file_list)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        file_path=self.file_list[index]
        image=Image.open(file_path)
        image=self.transform(image)
        return image