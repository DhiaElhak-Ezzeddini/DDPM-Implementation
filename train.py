import os 
import torch
import numpy as np 
from tqdm import tqdm 
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from Unet import Unet
from MNIST_Dataset import MNIST_DataSet
from Noise_Generation import LinearScheduler

def Download_Dataset(dest_path , split="train"):
    
    assert split in ["train" , "test"] , "split must be only 'train' or 'test' "
    is_train = (split == "train")
    
    if is_train : 
        if os.path.exists('./data/train') : 
            print("MNIST dataset is ready ")
            return None
        
    print("Downloading the MNIST dataset ...")
    
    dataset = MNIST(root=dest_path , split=is_train , download=True)
    
    if not os.path.exists(dest_path) : 
        os.mkdir(dest_path)
        
    base_dir = os.path.join(dest_path , split)
    os.makedirs(base_dir , exist_ok=True)
    
    for idx , (img,label) in enumerate(dataset):
        label_dir = os.path.join(base_dir , str(label))
        os.makedirs(label_dir , exist_ok=True)
        img_path = os.path.join(label_dir , f"{idx}.png")
        img.save(img_path)
        
    print(f"downloaded and saved {len(dataset)} images to '{base_dir}'")
    

def train(diffusion_config ,train_config ,model_config , device):
    
    scheduler = LinearScheduler(n_ts=diffusion_config["n_ts"], beta_s=diffusion_config["beta_s"] , beta_e=diffusion_config["beta_e"])
    
    ## Dataset 
    MNIST = MNIST_DataSet('train' , im_path="./data/train")
    mnist_dataloader = DataLoader(MNIST,train_config["batch_size"],shuffle=True,num_workers=4)
    
    model = Unet(model_config["im_channels"]).to(device)
    model.train()
    
    ## outputs directions 
    if not os.path.exists(train_config["task_name"]):
        os.mkdir(train_config["task_name"])
        
    if os.path.exists(os.path.join(train_config["task_name"] , train_config["ckpt_name"])): 
        print("loading checkpoint as found one")
        model.load_state_dict(torch.load(os.path.join(train_config["task_name"] , train_config["ckpt_name"]) , map_location=device))

    ### Training Params 
    optimizer = torch.optim.Adam(model.parameters() , lr=train_config["lr"])
    criterion = torch.nn.MSELoss()
    
    print("######################################################")
    print("############## Start Training The Model ##############")
    print("######################################################")
    
    for epoch in range(train_config["num_epochs"]):
        losses = []
        print(f"@@@@@ processing Epoch {epoch+1} @@@@@")
        for im in tqdm(mnist_dataloader):
            
            optimizer.zero_grad()
            ###print(f"length of the batch {len(im)}")
            im = im.float().to(device)
            
            ### Sample random noise 
            noise = torch.randn_like(im).to(device)
            
            ### Sample timestep 
            
            t = torch.randint(0,diffusion_config["n_ts"],(im.shape[0],)).to(device)
            
            ### Add noise to image according to timestep 
            
            noisy_im = scheduler.add_noise(im,noise,t,device)
            
            noise_pred = model(noisy_im,t)
            
            loss = criterion(noise_pred , noise)
            losses.append(loss)
            loss.backward()
            optimizer.step()

        print("Finished epoch:{} | Loss : {:.4f}".format(epoch+1,np.mean([loss.cpu().item() for loss in losses])))
        ## save the model state at each epoch 
        torch.save(model.state_dict() , f"./default/model_epoch_{epoch+1}.pth" )
        
    print("Done Training ....")

if __name__ == "__main__" : ### for trining in google colab replace it with " def execute(): "    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device : {device}")
    
    diffusion_config = {"n_ts" : 1000, "beta_s" : 0.0001 , "beta_e" : 0.02}
    
    train_config = {"task_name": '/content/drive/MyDrive/Colab_Notebooks/default',"batch_size" : 64,"num_epochs": 30 ,"num_samples" : 100,
                    "num_grid_rows" : 10,"lr": 0.0001} ## task_name made for training on Google Colab  
    
    model_config = {"im_channels" : 1,"im_size" : 28,"down_channels" : [32, 64, 128, 256],"mid_channels" : [256, 256, 128],
                        "down_sample" : [True, True, False],"time_emb_dim" : 128,"num_down_layers" : 2,"num_mid_layers" : 2,
                        "num_up_layers": 2,"num_heads" : 4}
    
    Download_Dataset("./data" , split="train")
    
    train(diffusion_config ,train_config ,model_config , device) 
    
        
        
    