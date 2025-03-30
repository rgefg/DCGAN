import torch
from torchvision import utils
from torch import nn
import torch.optim.adam
from torchvision import transforms  
from anidataset import anidataset
from torch.utils.data import DataLoader
#torchvision.datasets：内置数据集（如CIFAR10）常配合torchvison.transforms使用,可以把dataset理解为Dataset这一父基类的实例数据集。
# torch.utils.data:
# Dataset：定义数据的存储结构和单样本访问逻辑（需用户继承并实现__len__和__getitem__）。
# DataLoader：批量加载数据，支持多进程加速、随机打乱等。

class generator(nn.Module):
    def __init__(self, noise_size, output_channels):
        super(generator,self).__init__()
        self.main=nn.Sequential(
            nn.ConvTranspose2d(
            in_channels=noise_size,
            out_channels=1024,
            kernel_size=4,
            stride=1,
            padding=0,
            bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.ConvTranspose2d(
            in_channels=1024,
            out_channels=512,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(
            in_channels=512,
            out_channels=256,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(
            in_channels=256,
            out_channels=128,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(
            in_channels=128,
            out_channels=3,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False
            ),
            nn.Tanh()
        )
    def forward(self,x):
        x=x.reshape(x.size(0),-1,1,1)
        x=self.main(x)
        return x

class discriminator(nn.Module):
    def __init__(self,in_channels):
        super(discriminator,self).__init__()
        self.main=nn.Sequential(
        nn.Conv2d(in_channels=in_channels,out_channels=64,kernel_size=4,stride=2,padding=1,bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(in_channels=64,out_channels=128,kernel_size=4,stride=2,padding=1,bias=False),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(128,256,kernel_size=4,stride=2,padding=1,bias=False),
        nn.BatchNorm2d(256),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(256,512,kernel_size=4,stride=2,padding=1,bias=False),
        nn.BatchNorm2d(512),
        nn.LeakyReLU(0.2, inplace=True)
        )  
        self.linear=nn.Sequential(nn.Linear(in_features=512*4*4,out_features=1,bias=False))

    def forward(self,x):
        x=self.main(x)
        x=x.flatten(start_dim=1)
        return self.linear(x)
    
def weights_init(m):
    classname=m.__class__.__name__
    if classname.find('Conv')!=-1:
        nn.init.normal_(m.weight.data,0,0.02)
    elif classname.find('BatchNorm')!=-1:
        nn.init.normal_(m.weight.data,1,0.02)
        nn.init.constant_(m.bias.data,0)



if __name__=='__main__':
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fixed_noise = torch.randn(64, 100, 1, 1).to(device)
    out_dir='01-basics/feedforward_neural_network/GAN/image'
    print(torch.cuda.is_available())
    trans=transforms.Compose([transforms.Resize(64),transforms.CenterCrop(64),transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    dataset=anidataset('/mnt/e/anime/images/',trans)
    dataloader=DataLoader(dataset,batch_size=64,shuffle=True)
    noise_size=100
    output_channels=3
    netG=generator(noise_size=noise_size,output_channels=output_channels).to(device)
    netG.apply(weights_init)
    netD=discriminator(output_channels).to(device)
    netD.apply(weights_init)
    opG=torch.optim.Adam(netG.parameters(),lr=0.002,betas=(0.5,0.999))              #betas接受元组（动量，衰减率）
    opD=torch.optim.Adam(netD.parameters(),lr=0.002,betas=(0.5,0.999))
    loss=nn.BCEWithLogitsLoss()

    n_epoch=300
    for epoch in range(n_epoch):
        for i,data in enumerate(dataloader):
            data=data.to(device)
            opD.zero_grad()
            output=netD(data)
            real_loss=loss(output,torch.ones_like(output))
            noise1=torch.randn(64,noise_size,1,1,device=device).to(device)
            fake_img1=netG(noise1)
            fake_output1=netD(fake_img1)
            fake_loss=loss(fake_output1,torch.zeros_like(fake_output1))
            D_loss=fake_loss+real_loss
            D_loss.backward()
            torch.nn.utils.clip_grad_norm_(netD.parameters(), max_norm=5)
            opD.step()

            opG.zero_grad()
            noise2=torch.randn(64,noise_size,1,1).to(device)       #重新采样noise2，不复用noise1，固定了判别器的参数
            fake_img2=netG(noise2)
            fake_output2=netD(fake_img2)
            G_loss=loss(fake_output2,torch.ones_like(fake_output2))
            G_loss.backward()
            torch.nn.utils.clip_grad_norm_(netG.parameters(), max_norm=5)
            opG.step()

            
            if epoch % 50 == 0:
                torch.save(netG.state_dict(), f'{out_dir}/netG_epoch_{epoch}.pth')
                torch.save(netD.state_dict(), f'{out_dir}/netD_epoch_{epoch}.pth')
            if i%25==0:
                print('[%d/%d][%d/%d] loss_d: %.4f loss_g:%.4f'%(epoch,n_epoch,i,len(dataloader),D_loss.item(),G_loss.item()))
                if epoch%10==0:
                    utils.save_image(data,'%s/real.png' % out_dir,normalize=True)
                    fake=netG(fixed_noise)
                    utils.save_image(fake.detach(),'%s/fake_epoch_%d.png'%(out_dir,epoch),normalize=True)
