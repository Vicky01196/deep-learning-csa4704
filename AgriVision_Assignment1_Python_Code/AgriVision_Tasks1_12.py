
"""
CSA47 Deep Learning CO1 Assignment 1 — AgriVision
Student: Vigneshwaran B | Register No.: 192324022

This script:
1) Reproduces numerical examples in Tasks 2–4 and capacity plot in Task 6.
2) Provides an optional real image-classification experiment on a folder dataset
   arranged as ImageFolder (e.g. PlantVillage subset: Healthy/Early Blight/Late Blight).
3) Reports metrics and confusion matrix; does NOT fabricate experimental outcomes.

Install:
pip install numpy matplotlib scikit-learn torch torchvision

Numerical section:
python AgriVision_Tasks1_12.py

Image experiment:
python AgriVision_Tasks1_12.py --data ./PlantVillage3Class --epochs 5
Folder structure:
PlantVillage3Class/
  train/Healthy/*.jpg
  train/Early_Blight/*.jpg
  train/Late_Blight/*.jpg
  val/Healthy/*.jpg
  val/Early_Blight/*.jpg
  val/Late_Blight/*.jpg
  test/Healthy/*.jpg
  test/Early_Blight/*.jpg
  test/Late_Blight/*.jpg

Use a documented, stratified split and record the dataset source/version.
"""
import os, argparse, time, math
import numpy as np
import matplotlib.pyplot as plt

SEED=42
np.random.seed(SEED)

def numerical_tasks():
    print("\\n=== Task 2: SVD/PCA worked example ===")
    B=np.array([[2.,0.],[0.,1.]])
    U,s,Vt=np.linalg.svd(B)
    rank1=U[:,:1] @ np.diag(s[:1]) @ Vt[:1,:]
    err=np.linalg.norm(B-rank1,ord='fro')
    print("Singular values:",np.round(s,4))
    print("Rank-1 approximation:\\n",np.round(rank1,4))
    print("Frobenius reconstruction error:",round(float(err),4))
    print("Rank-1 energy retained:",round(float(s[0]**2/(s**2).sum()),4))

    print("\\n=== Task 3: Entropy / cross-entropy / KL ===")
    p=np.array([.5,.3,.2]); q=np.array([.4,.4,.2])
    H=-np.sum(p*np.log2(p)); CE=-np.sum(p*np.log2(q)); KL=np.sum(p*np.log2(p/q))
    print("H(p) =",round(float(H),4),"bits")
    print("H(p,q) =",round(float(CE),4),"bits")
    print("KL(p||q) =",round(float(KL),4),"bits")
    print("KL(q||p) =",round(float(np.sum(q*np.log2(q/p))),4),"bits (reverse direction)")

    print("\\n=== Task 4: Numerically stable softmax ===")
    z=np.array([1000.,999.,995.]); shifted=z-z.max()
    probs=np.exp(shifted)/np.exp(shifted).sum()
    print("Original logits:",z)
    print("Shifted logits:",shifted)
    print("Stable probabilities:",np.round(probs,4))
    print("Covariance condition number example:",np.linalg.cond(np.diag([4.,1.,1e-4])))

    print("\\n=== Task 6: Capacity plot (ILLUSTRATIVE values; replace with measured values) ===")
    capacity=np.arange(1,6)
    train=np.array([.28,.18,.10,.04,.01])
    valid=np.array([.31,.22,.16,.19,.27])
    gap=valid-train; opt=np.argmin(valid)
    plt.figure(figsize=(8,5))
    plt.plot(capacity,train,'o-',label='Training error')
    plt.plot(capacity,valid,'o-',label='Validation error')
    plt.axvspan(.7,2.0,alpha=.12,label='Underfitting region')
    plt.axvspan(3.7,5.3,alpha=.12,label='Overfitting region')
    plt.scatter(capacity[opt],valid[opt],marker='*',s=180,label='Optimal capacity')
    plt.annotate(f'Gap at C4 = {gap[3]:.4f}',xy=(4,valid[3]),xytext=(2.6,.30),
                 arrowprops={'arrowstyle':'->'})
    plt.xticks(capacity,[f'C{i}' for i in capacity])
    plt.xlabel('Model capacity'); plt.ylabel('Error')
    plt.title('AgriVision: Training and Validation Error vs Capacity')
    plt.grid(alpha=.25); plt.legend(); plt.tight_layout()
    plt.savefig('AgriVision_capacity_plot.png',dpi=180)
    print("Generalisation gaps:",np.round(gap,4))
    print("Illustrative optimal capacity: C",capacity[opt],sep='')
    print("Saved AgriVision_capacity_plot.png")
    print("NOTE: Values are examples from the written report, not model-training results.")

def run_image_experiment(root,epochs=5,image_size=64,batch_size=32):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision import datasets,transforms
    from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    print("\\n=== Image experiment ===")
    print("Device:",device)
    # ImageFolder expects subfolders named by class.
    tf_train=transforms.Compose([
        transforms.Resize((image_size,image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor()])
    tf_eval=transforms.Compose([
        transforms.Resize((image_size,image_size)),
        transforms.ToTensor()])
    train_ds=datasets.ImageFolder(os.path.join(root,'train'),transform=tf_train)
    val_ds=datasets.ImageFolder(os.path.join(root,'val'),transform=tf_eval)
    test_ds=datasets.ImageFolder(os.path.join(root,'test'),transform=tf_eval)
    print("Classes:",train_ds.classes)
    print("Counts train/val/test:",len(train_ds),len(val_ds),len(test_ds))
    if train_ds.classes!=val_ds.classes or train_ds.classes!=test_ds.classes:
        raise ValueError("Train/val/test class folders must match.")
    train_dl=DataLoader(train_ds,batch_size=batch_size,shuffle=True,num_workers=0)
    val_dl=DataLoader(val_ds,batch_size=batch_size,shuffle=False,num_workers=0)
    test_dl=DataLoader(test_ds,batch_size=batch_size,shuffle=False,num_workers=0)

    # Small CNN baseline
    class LeafCNN(nn.Module):
        def __init__(self,nclass):
            super().__init__()
            self.features=nn.Sequential(
                nn.Conv2d(3,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
                nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
                nn.Conv2d(32,64,3,padding=1),nn.ReLU(),
                nn.AdaptiveAvgPool2d((1,1)))
            self.classifier=nn.Linear(64,nclass)
        def forward(self,x): return self.classifier(self.features(x).flatten(1))
    model=LeafCNN(len(train_ds.classes)).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
    criterion=nn.CrossEntropyLoss()
    best_val=float('inf'); best_state=None
    for epoch in range(1,epochs+1):
        model.train(); loss_sum=correct=n=0
        for x,y in train_dl:
            x,y=x.to(device),y.to(device)
            optimizer.zero_grad(); logits=model(x); loss=criterion(logits,y)
            loss.backward(); optimizer.step()
            loss_sum+=loss.item()*len(y); correct+=(logits.argmax(1)==y).sum().item(); n+=len(y)
        model.eval(); vl=vc=vn=0
        with torch.no_grad():
            for x,y in val_dl:
                x,y=x.to(device),y.to(device); logits=model(x); loss=criterion(logits,y)
                vl+=loss.item()*len(y); vc+=(logits.argmax(1)==y).sum().item(); vn+=len(y)
        print(f"Epoch {epoch}/{epochs}: train_loss={loss_sum/n:.4f}, train_acc={correct/n:.4f}, val_loss={vl/vn:.4f}, val_acc={vc/vn:.4f}")
        if vl/vn<best_val:
            best_val=vl/vn; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state); model.eval()
    y_true=[]; y_pred=[]
    t0=time.perf_counter()
    with torch.no_grad():
        for x,y in test_dl:
            pred=model(x.to(device)).argmax(1).cpu().numpy()
            y_pred.extend(pred); y_true.extend(y.numpy())
    elapsed=time.perf_counter()-t0
    print("\\nTest accuracy:",accuracy_score(y_true,y_pred))
    print("Inference seconds for entire test set:",round(elapsed,4))
    print("Classification report:\\n",classification_report(y_true,y_pred,target_names=test_ds.classes,zero_division=0))
    print("Confusion matrix:\\n",confusion_matrix(y_true,y_pred))
    torch.save({'model_state':model.state_dict(),'classes':train_ds.classes,
                'image_size':image_size},'AgriVision_CNN.pth')
    print("Saved model: AgriVision_CNN.pth")
    print("For mobile latency/model size, benchmark on the actual target phone; desktop timing is not phone evidence.")

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',default=None,help='ImageFolder root containing train/val/test directories')
    parser.add_argument('--epochs',type=int,default=5)
    parser.add_argument('--image-size',type=int,default=64)
    parser.add_argument('--batch-size',type=int,default=32)
    args=parser.parse_args()
    numerical_tasks()
    if args.data:
        run_image_experiment(args.data,args.epochs,args.image_size,args.batch_size)
    else:
        print("\\nTo run real image training, pass --data PATH (see README).")
