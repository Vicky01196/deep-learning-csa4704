import torch, numpy as np
from torch import nn
from torchvision import datasets,transforms
from torch.utils.data import DataLoader,Subset
torch.manual_seed(42); np.random.seed(42); dev='cuda' if torch.cuda.is_available() else 'cpu'
ds=datasets.FashionMNIST('./data',train=True,download=True,transform=transforms.ToTensor())
tr=Subset(ds,range(50000)); va=Subset(ds,range(50000,60000))
class Net(nn.Module):
 def __init__(self):
  super().__init__(); self.net=nn.Sequential(nn.Flatten(),nn.Linear(784,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU(),nn.Linear(128,10))
 def forward(self,x): return self.net(x)
def run(kind,lr,batch,epochs=8):
 torch.manual_seed(42); m=Net().to(dev)
 opts={'SGD':torch.optim.SGD(m.parameters(),lr=lr),'Momentum':torch.optim.SGD(m.parameters(),lr=lr,momentum=.9),
 'RMSprop':torch.optim.RMSprop(m.parameters(),lr=lr),'Adam':torch.optim.Adam(m.parameters(),lr=lr)}
 opt=opts[kind]; ce=nn.CrossEntropyLoss()
 for _ in range(epochs):
  m.train()
  for x,y in DataLoader(tr,batch_size=batch,shuffle=True):
   x,y=x.to(dev),y.to(dev); opt.zero_grad(); l=ce(m(x),y); l.backward(); opt.step()
 m.eval(); vl=c=n=0
 with torch.no_grad():
  for x,y in DataLoader(va,256):
   x,y=x.to(dev),y.to(dev); z=m(x); vl+=ce(z,y).item()*len(y); c+=(z.argmax(1)==y).sum().item(); n+=len(y)
 return dict(optimizer=kind,lr=lr,batch=batch,val_loss=vl/n,val_accuracy=c/n)
if __name__=='__main__':
 for o in ['SGD','Momentum','RMSprop','Adam']:
  for lr in [1e-4,1e-3,1e-2]:
   for b in [32,128]: print(run(o,lr,b))
