import torch, time, numpy as np
from torch import nn
from torchvision import datasets,transforms
from torch.utils.data import DataLoader,Subset
torch.manual_seed(42); np.random.seed(42); dev='cuda' if torch.cuda.is_available() else 'cpu'
ds=datasets.FashionMNIST('./data',train=True,download=True,transform=transforms.ToTensor())
tr=Subset(ds,range(50000)); va=Subset(ds,range(50000,60000))
class CustomCNN(nn.Module):
 def __init__(self,D=2,F=8,R=28):
  super().__init__(); self.R=R; layers=[]; ci=1
  for i in range(D):
   co=F*(2**(i//2)); layers += [nn.Conv2d(ci,co,3,padding=1),nn.ReLU()]
   if i%2==1: layers += [nn.MaxPool2d(2)]
   ci=co
  self.features=nn.Sequential(*layers); self.gap=nn.AdaptiveAvgPool2d(1); self.fc=nn.Linear(ci,10)
 def forward(self,x):
  if x.shape[-1]!=self.R: x=torch.nn.functional.interpolate(x,size=(self.R,self.R))
  return self.fc(self.gap(self.features(x)).flatten(1))
def count_macs(m,shape):
 total=[0]
 def hook(layer,inp,out):
  if isinstance(layer,nn.Conv2d):
   b,c,h,w=out.shape; kh,kw=layer.kernel_size
   total[0]+=b*h*w*c*(layer.in_channels//layer.groups)*kh*kw
 hooks=[l.register_forward_hook(hook) for l in m.modules()]
 m(torch.zeros(shape).to(dev))
 for h in hooks:h.remove()
 return total[0]
def run(D,F,R,epochs=2):
 m=CustomCNN(D,F,R).to(dev); opt=torch.optim.Adam(m.parameters(),lr=1e-3); ce=nn.CrossEntropyLoss()
 for _ in range(epochs):
  m.train()
  for x,y in DataLoader(tr,128,shuffle=True):
   x,y=x.to(dev),y.to(dev); opt.zero_grad(); l=ce(m(x),y); l.backward(); opt.step()
 m.eval(); c=n=0
 with torch.no_grad():
  for x,y in DataLoader(va,256): c+=(m(x.to(dev)).argmax(1).cpu()==y).sum().item(); n+=len(y)
 x=torch.randn(1,1,R,R).to(dev)
 if dev=='cuda': torch.cuda.synchronize()
 t=time.perf_counter()
 with torch.no_grad():
  for _ in range(100): m(x)
 if dev=='cuda': torch.cuda.synchronize()
 return {'D':D,'F':F,'R':R,'params':sum(p.numel() for p in m.parameters()),'MACs':count_macs(m,(1,1,R,R)),
 'size_MB':sum(p.numel()*p.element_size() for p in m.parameters())/1e6,'latency_ms':(time.perf_counter()-t)*10,'val_accuracy':c/n}
if __name__=='__main__':
 for cfg in [(2,8,28),(4,16,28),(4,16,56)]: print(run(*cfg))
