import torch 


class LinearScheduler:
    def __init__(self , n_ts , beta_s , beta_e):
        super().__init__()
        
        self.n_ts = n_ts
        self.beta_s = beta_s
        self.beta_e = beta_e
        
        self.betas = torch.linspace(beta_s , beta_e , n_ts)
        
        self.alphas = 1. - self.betas
        self.sqrt_alphas = torch.sqrt(self.alphas)
        
        self.alphas_bar  = torch.cumprod(self.alphas , dim=0)
        self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar)
        

    def add_noise(self,x0,noise,t,device): ## adding noise to the original image with a given time step 
        x0_shape = x0.shape

        batch_size = x0_shape[0]
        for _ in range(len(x0_shape)-1):
            alphas_bar = self.alphas_bar[t.cpu()].reshape(batch_size,1,1,1)
            sqrt_alphas_bar = self.sqrt_alphas_bar[t.cpu()].reshape(batch_size,1,1,1)
    
        alphas_bar = alphas_bar.to(device)
        sqrt_alphas_bar = sqrt_alphas_bar.to(device)
        return sqrt_alphas_bar*x0 + torch.sqrt(1-alphas_bar)*noise ## starting from x0 and returning xt (Noisy x0 at a given time step t)

    def sample_prev_time_step(self,xt,noise_theta,t):

        x0 = (xt - (torch.sqrt(1-self.alphas_bar[t]))) / self.sqrt_alphas_bar[t] ## from the previous formula in the add_noise function
        x0 = torch.clamp(x0 , min=-1. , max=1)
    
    
        mean_theta = (xt - ((1-self.alphas[t])/torch.sqrt(1 - self.alphas_bar[t])) * noise_theta ) / self.sqrt_alphas[t]
    
        if t==0:
            return mean_theta , x0
        else : 
            sigma = ( ((1-self.alphas[t])*(1-self.alphas_bar[t-1]))/(1-self.alphas_bar[t]) ) ** 0.5
            z = torch.randn(xt.shape).to(xt.device)
            return (mean_theta + sigma*z) , x0 

