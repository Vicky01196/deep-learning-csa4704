import torch, numpy as np, matplotlib.pyplot as plt
from torch import nn
from torchvision import datasets,transforms
from torch.utils.data import DataLoader,Subset
torch.manual_seed(42); np.random.seed(42); dev='cuda' if torch.cuda.is_available() else 'cpu'
ds=datasets.FashionMNIST('./data',train=True,download=True,transform=transforms.ToTensor())
tr=Subset(ds,range(1000)); va=Subset(ds,range(50000,60000))
def dl(d,sh=False): return DataLoader(d,128,shuffle=sh)
class MLP(nn.Module):
 def __init__(self,drop=0):
  super().__init__(); self.net=nn.Sequential(nn.Flatten(),nn.Linear(784,512),nn.ReLU(),nn.Dropout(drop),nn.Linear(512,256),nn.ReLU(),nn.Dropout(drop),nn.Linear(256,10))
 def forward(self,x): return self.net(x)
def run(name,wd=0,drop=0,aug=False,patience=None,epochs=30):
 m=MLP(drop).to(dev); opt=torch.optim.Adam(m.parameters(),lr=1e-3,weight_decay=wd); ce=nn.CrossEntropyLoss()
 h={'tr_loss':[],'va_loss':[],'tr_acc':[],'va_acc':[]}; best=1e9; wait=0; bestep=0
 for ep in range(epochs):
  m.train(); tl=tc=n=0
  for x,y in dl(tr,True):
   if aug: x=torch.roll(x,(np.random.randint(-2,3),np.random.randint(-2,3)),(2,3))
   x,y=x.to(dev),y.to(dev); opt.zero_grad(); z=m(x); l=ce(z,y); l.backward(); opt.step()
   tl+=l.item()*len(y); tc+=(z.argmax(1)==y).sum().item(); n+=len(y)
  m.eval(); vl=vc=vn=0
  with torch.no_grad():
   for x,y in dl(va):
    x,y=x.to(dev),y.to(dev); z=m(x); l=ce(z,y); vl+=l.item()*len(y); vc+=(z.argmax(1)==y).sum().item(); vn+=len(y)
  for k,v in [('tr_loss',tl/n),('va_loss',vl/vn),('tr_acc',tc/n),('va_acc',vc/vn)]: h[k].append(v)
  if vl/vn<best: best=vl/vn; bestep=ep+1; wait=0
  else: wait+=1
  if patience and wait>=patience: break
 print(name,'best validation epoch=',bestep,'weight norm=',sum(p.norm().item()**2 for p in m.parameters())**.5)
 return h
if __name__=='__main__':
 configs={'baseline':{},'L2':{'wd':1e-4},'dropout':{'drop':.4},'early_stop':{'patience':5},'augmentation':{'aug':True},'combined':{'wd':1e-4,'drop':.3,'aug':True,'patience':5}}
 allh={k:run(k,**v) for k,v in configs.items()}
 plt.figure()
 for k,h in allh.items(): plt.plot(h['va_loss'],label=k)
 plt.xlabel('Epoch'); plt.ylabel('Validation loss'); plt.legend(); plt.tight_layout(); plt.savefig('Q3_val_loss.png')
