import torch, numpy as np
from torch import nn
from torchvision import datasets,transforms
from torch.utils.data import DataLoader,Subset
torch.manual_seed(42); np.random.seed(42); dev='cuda' if torch.cuda.is_available() else 'cpu'
ds=datasets.FashionMNIST('./data',train=True,download=True,transform=transforms.ToTensor())
tr=Subset(ds,range(50000)); va=Subset(ds,range(50000,60000))
class Deep(nn.Module):
 def __init__(self,act='sigmoid'):
  super().__init__(); self.h=nn.ModuleList(); d=784; self.act=act
  for _ in range(8): self.h.append(nn.Linear(d,128)); d=128
  self.out=nn.Linear(d,10)
  for l in list(self.h)+[self.out]:
   nn.init.kaiming_normal_(l.weight) if act=='relu' else nn.init.xavier_uniform_(l.weight)
 def forward(self,x):
  x=x.flatten(1)
  for l in self.h: x=l(x); x=torch.sigmoid(x) if self.act=='sigmoid' else torch.relu(x)
  return self.out(x)
def run(act,epochs=3):
 m=Deep(act).to(dev); opt=torch.optim.Adam(m.parameters(),lr=1e-3); ce=nn.CrossEntropyLoss(); norms=[]
 for ep in range(epochs):
  m.train()
  for x,y in DataLoader(tr,128,shuffle=True):
   x,y=x.to(dev),y.to(dev); opt.zero_grad(); z=m(x); ce(z,y).backward()
   norms.append([l.weight.grad.norm().item() for l in m.h]); opt.step()
 m.eval(); c=n=vl=0
 with torch.no_grad():
  for x,y in DataLoader(va,256):
   z=m(x.to(dev)); vl+=ce(z,y.to(dev)).item()*len(y); c+=(z.argmax(1).cpu()==y).sum().item(); n+=len(y)
 return {'activation':act,'val_loss':vl/n,'val_accuracy':c/n,'layer_gradient_norms':np.mean(norms,axis=0).tolist(),'params':sum(p.numel() for p in m.parameters())}
if __name__=='__main__': print('Original',run('sigmoid')); print('Redesign',run('relu'))
