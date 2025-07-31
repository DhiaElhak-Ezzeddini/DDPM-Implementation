import os 
import torch
import torchvision 
from torchvision.utils import make_grid
from tqdm import tqdm

from Unet import Unet 
from Noise_Generation import LinearScheduler

def sample(model , scheduler ,train_config , model_config , diffusion_config,device):
    
    xt = torch.randn((train_config["num_samples"],
                      model_config["im_channels"],
                      model_config["im_size"],
                      model_config["im_size"])).to(device) ### 
    
    for i in tqdm(reversed(range(diffusion_config["n_ts"]))):
        
        noise_pred = model(xt,torch.as_tensor(i).unsqueeze(0).to(device))

        xt , x0_pred = scheduler.sample_prev_time_step(xt,noise_pred,torch.as_tensor(i).to(device))
        
        ims = torch.clamp(xt,min=-1,max=1).detach().cpu()
        ims = (ims+1)/2
        grid = make_grid(ims,nrow=train_config["num_grid_rows"])
        img = torchvision.transforms.ToPILImage()(grid)
        if not os.path.exists(os.path.join(train_config['task_name'] , "samples")) : 
            os.mkdir(os.path.join(train_config['task_name'] , "samples"))
        img.save(os.path.join(train_config['task_name'] , "samples" , "x0_{}.png".format(i)))
        img.close()
        
def infer() : 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    diffusion_config = {"n_ts" : 1000, "beta_s" : 0.0001 , "beta_e" : 0.02}
    
    train_config = {"task_name": 'default',"batch_size" : 64,"num_epochs": 30 ,"num_samples" : 100,"num_grid_rows" : 10,
                        "lr": 0.0001}
    
    model_config = {"im_channels" : 1,"im_size" : 28,"down_channels" : [32, 64, 128, 256],"mid_channels" : [256, 256, 128],
                        "down_sample" : [True, True, False],"time_emb_dim" : 128,"num_down_layers" : 2,"num_mid_layers" : 2,
                        "num_up_layers": 2,"num_heads" : 4}
    
    model = Unet(model_config["im_channels"]).to(device)
    model.load_state_dict(torch.load(os.path.join(train_config['task_name'],"model_epoch_30.pth"),map_location=device)) ## state of the last epoch of training 
    
    
    model.eval()
    scheduler = LinearScheduler(n_ts=diffusion_config["n_ts"], beta_s=diffusion_config["beta_s"] , beta_e=diffusion_config["beta_e"])
    
    with torch.no_grad():

        sample(model , scheduler , train_config , model_config , diffusion_config , device)  
        
        
if __name__ == '__main__' : 
    infer()      

        
        