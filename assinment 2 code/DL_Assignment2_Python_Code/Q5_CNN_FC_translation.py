import torch, numpy as np, matplotlib.pyplot as plt
from torch import nn
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
torch.manual_seed(42); np.random.seed(42); dev='cuda' if torch.cuda.is_available() else 'cpu'
tr=datasets.FashionMNIST('./data',train=True,download=True,transform=transforms.ToTensor())
te=datasets.FashionMNIST('./data',train=False,download=True,transform=transforms.ToTensor())
class FC(nn.Module):
 def __init__(self): super().__init__(); self.net=nn.Sequential(nn.Flatten(),nn.Linear(784,128),nn.ReLU(),nn.Linear(128,10))
 def forward(self,x): return self.net(x)
class CNN(nn.Module):
 def __init__(self):
  super().__init__(); self.c1=nn.Conv2d(1,32,3,padding=1); self.c2=nn.Conv2d(32,64,3,padding=1); self.pool=nn.MaxPool2d(2); self.fc=nn.Linear(64*7*7,10)
 def forward(self,x): x=self.pool(torch.relu(self.c1(x))); x=self.pool(torch.relu(self.c2(x))); return self.fc(x.flatten(1))
def train_eval(m,epochs=3):
 m=m.to(dev); opt=torch.optim.Adam(m.parameters(),lr=1e-3); ce=nn.CrossEntropyLoss()
 for _ in range(epochs):
  m.train()
  for x,y in DataLoader(tr,128,shuffle=True):
   x,y=x.to(dev),y.to(dev); opt.zero_grad(); l=ce(m(x),y); l.backward(); opt.step()
 def acc(shift=0):
  m.eval(); c=n=0
  with torch.no_grad():
   for x,y in DataLoader(te,256):
    if shift: x=torch.roll(x,(shift,shift),(2,3))
    c+=(m(x.to(dev)).argmax(1).cpu()==y).sum().item(); n+=len(y)
  return c/n
 return acc(),acc(2)
if __name__=='__main__':
 for m in [FC(),CNN()]:
  a,b=train_eval(m); print(type(m).__name__,'params=',sum(p.numel() for p in m.parameters()),'test=',a,'shifted=',b)
 m=CNN(); w=m.c1.weight.detach().cpu(); fig,axs=plt.subplots(4,8)
 for i,ax in enumerate(axs.flat): ax.imshow(w[i,0],cmap='gray'); ax.axis('off')
 plt.savefig('Q5_filters.png')
 x,_=te[0]; f=torch.relu(m.c1(x[None])).detach()[0]; fig,axs=plt.subplots(4,8)
 for i,ax in enumerate(axs.flat): ax.imshow(f[i],cmap='gray'); ax.axis('off')
 plt.savefig('Q5_featuremaps.png')
