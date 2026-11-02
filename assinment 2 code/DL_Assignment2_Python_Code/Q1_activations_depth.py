import torch, numpy as np
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import precision_score, recall_score
SEED=42; torch.manual_seed(SEED); np.random.seed(SEED)
dev='cuda' if torch.cuda.is_available() else 'cpu'
ds=datasets.FashionMNIST('./data',train=True,download=True,transform=transforms.ToTensor())
tr=Subset(ds,range(50000)); va=Subset(ds,range(50000,60000))
def loader(d,sh=False): return DataLoader(d,batch_size=128,shuffle=sh)
class Model(nn.Module):
 def __init__(self,depth,act,init='xavier',width=64):
  super().__init__(); self.layers=nn.ModuleList(); self.act=act; d=784
  for _ in range(depth): self.layers.append(nn.Linear(d,width)); d=width
  self.out=nn.Linear(d,10)
  for l in list(self.layers)+[self.out]:
   if init=='bad': nn.init.normal_(l.weight,std=1.5)
   elif init=='he' and act=='relu': nn.init.kaiming_normal_(l.weight)
   else: nn.init.xavier_uniform_(l.weight)
 def forward(self,x):
  x=x.flatten(1)
  for l in self.layers:
   x=l(x)
   x=torch.sigmoid(x) if self.act=='sigmoid' else torch.tanh(x) if self.act=='tanh' else torch.relu(x)
  return self.out(x)
def experiment(depth,act,init='xavier',epochs=3):
 m=Model(depth,act,init).to(dev); opt=torch.optim.Adam(m.parameters(),lr=1e-3); loss=nn.CrossEntropyLoss()
 for ep in range(epochs):
  m.train()
  for x,y in loader(tr,True):
   x,y=x.to(dev),y.to(dev); opt.zero_grad(); z=m(x); loss(z,y).backward()
   gn=[l.weight.grad.norm().item() for l in m.layers]; opt.step()
 m.eval(); ys=[]; ps=[]; dead=[]
 with torch.no_grad():
  for x,y in loader(va):
   x=x.to(dev); pred=m(x).argmax(1).cpu(); ys.extend(y.numpy()); ps.extend(pred.numpy())
   if act=='relu':
    for l in m.layers: dead.append((l(x.flatten(1))<=0).float().mean().item())
 return dict(depth=depth,activation=act,init=init,layer_grad_norms=gn,
  dead_relu_fraction=float(np.mean(dead)) if dead else None,
  accuracy=float(np.mean(np.array(ys)==np.array(ps))),
  precision_macro=precision_score(ys,ps,average='macro',zero_division=0),
  recall_macro=recall_score(ys,ps,average='macro',zero_division=0))
if __name__=='__main__':
 for d,i in [(2,'xavier'),(8,'xavier'),(8,'bad')]:
  for a in ['sigmoid','tanh','relu']: print(experiment(d,a,i))
