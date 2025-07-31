import glob
import os

import torchvision 
from PIL import Image
from tqdm import tqdm 

from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

class MNIST_DataSet(Dataset):
    def __init__(self,split,im_path,im_ext='png'):
        self.split = split
        self.im_ext = im_ext
        self.images , self.labels = self.load_images(im_path)
        
    def load_images(self,im_path):
        assert os.path.exists(im_path) , "imp path {} not found ".format(im_path)
        ims = []
        labels = []
        for d_name in tqdm(os.listdir(im_path)):
            for fname in glob.glob(os.path.join(im_path,d_name,'*.{}'.format(self.im_ext))):
                ims.append(fname)
                labels.append(int(d_name))
        
        print("found {} images for split {} ".format(len(ims) , self.split))
        return ims , labels
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,index):
        im = Image.open(self.images[index])
        im_tensor = torchvision.transforms.ToTensor()(im)
        
        
        ## convert to -1  1 range 
        im_tensor = (2*im_tensor) -1 
        return im_tensor