from torchvision import models
import torch
import torchattacks
import torch.nn as nn
import dataset
from collections import Counter
import numpy as np
import pdb
import matplotlib.pyplot as plt
from cifar10_models.vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn

class Normalization(nn.Module):
    def __init__(self, mean, std,device):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)
        self.mean = self.mean.to(device, torch.float)
        self.std = self.std.to(device, torch.float)

    def forward(self, img):
        return (img - self.mean) / self.std

class LayerResult:
    def __init__(self,layers,layer_index):
        self.hook = layers[layer_index].register_forward_hook(self.hook_fn)

    def hook_fn(self,module,input,output):
        self.features = output

    def unregister_forward_hook(self):
        self.hook.remove()

def forward_pre_hook(module,input,output):
    # print ("hook")
    # attention = attention.reshape([1,-1,1,1])

    return output* attention.reshape([1,-1,1,1])


class modified_vgg(nn.Module):
    def __init__(self,model,layer_idx,idx_list,device):
        super(modified_vgg,self).__init__()
        self.vgg = model
        self.normalization =  Normalization([0.485,0.456,0.406],[0.229,0.224,0.225],device)
        self.idx_list = idx_list
        self.features = self.vgg[1].features
        self.classifier = self.vgg[1].classifier
        self.layer_idx = layer_idx

    def forward(self,x):
        out = self.normalization(x)
        for k,layer in enumerate(self.features):
            out = layer(out)
            # if k == self.layer_idx :
            out[:,idx_list[k],:,:] = out[:,idx_list[k],:,:] * 0
      
                # out = out * 0 
                

            # if isinstance(layer,nn.Conv2d):
            #     pass
            # else :
            

        
        N,C,H,W = out.shape
        out = out.reshape([N,-1])
        out = self.classifier(out)
        return out


train_set,test_set = dataset.CIFAR10(normalize=False)
num_classes = 10
batch_size = 1
test_loader = torch.utils.data.DataLoader(test_set,batch_size = batch_size,shuffle = True)
# vgg = models.vgg19(pretrained=True)
# vgg = nn.Sequential( Normalization([0.485,0.456,0.406],[0.229,0.224,0.225]),vgg)
# vgg = vgg.eval()


delete_percent = 0.25
select_number = []

# device = 'cpu'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


success_layer = Counter()
attack_success = 0 
percent = 10/5

vgg = vgg13_bn(pretrained=True)
vgg = nn.Sequential(
    Normalization([0.485,0.456,0.406],[0.229,0.224,0.225],device),
    vgg
    )

idx_list = [[] for i in range(len(vgg[1].features))]

for i,(x,y) in enumerate(test_loader):

    vgg = vgg13_bn(pretrained=True)
    vgg = nn.Sequential(
        Normalization([0.485,0.456,0.406],[0.229,0.224,0.225],device),
        vgg
    )
    vgg = vgg.eval()
    vgg = vgg.to(device)
    attack = torchattacks.PGD(vgg, eps = 4/255)     
    x,y = x.to(device),y.to(device)
    for param in vgg.parameters():
        param.requires_grad = False

    # x_adv = attack(x,y)
    # z_clean = vgg(x)
    # z_adv = vgg(x_adv)
    # pdb.set_trace()

    # for k in range(len(vgg[1].features)-10):

    for k in range(31,32,1):


        # if isinstance(vgg[1].features[k],nn.ReLU):
        if isinstance(vgg[1].features[k],nn.Conv2d):
        # if isinstance(vgg[1].features[k],nn.ReLU):
            result = LayerResult(vgg[1].features,k)
            z_clean = vgg(x)
            activations_clean = result.features
            x_adv = attack(x,y)
            activations_adv = result.features

            z_adv = vgg(x_adv)

            diff_adv = torch.square(activations_clean.squeeze() - activations_adv.squeeze())
            
            diff_adv = torch.sum(diff_adv,dim = (1,2))
            
            N = len(diff_adv)
            idx = (-diff_adv).argsort()
            # idx = np.random.choice(range(len(diff_adv)), len(diff_adv), replace=False)
            
            attention = torch.ones_like(diff_adv)

            attention[idx[:int(N//percent)]] = 0
            idx_list[k] = idx[:int(N//percent)]
            hook = vgg[1].features[k].register_forward_hook(forward_pre_hook)
            
            z_after = vgg(x_adv)
            z_clean = vgg(x)
            # unregister forward hook
            hook.remove()
            # if k == 31 :
            #     idx_list = idx[:int(N//percent)]
            #     print (idx_list)
            #     break
            


            print (k, y,z_clean.argmax(dim = 1),z_adv.argmax(dim = 1),z_after.argmax(dim = 1))
            if (z_clean.argmax(dim = 1) != z_adv.argmax(dim = 1) and (z_clean.argmax(dim = 1) == z_after.argmax(dim = 1)) and z_clean.argmax(dim = 1) == y):
                success_layer[k] += 1

    if (z_clean.argmax(dim = 1) != z_adv.argmax(dim = 1)):
        attack_success += 1


    # modified_model = modified_vgg(vgg,k,idx_list,device)
    # attack_modified = torchattacks.PGD(modified_model, eps = 8/255)  
    # # x_adv_modified = attack_modified(x,y)
    # z_after_modified = modified_model(x_adv)

    # print (y.item(),z_after.argmax().item(),z_after_modified.argmax().item())

    # pdb.set_trace()
    if i == 1000:
        break


Xs = []
Ys = []
Xs.append(-1)
Ys.append(attack_success)
for k in success_layer.keys():
    Xs.append(k)
    Ys.append(success_layer[k])

plt.bar(Xs,Ys)
plt.savefig('./fig/bar_%.2f'%(1/percent) + '.png')
plt.close()

pdb.set_trace()


vgg = vgg13_bn(pretrained=True)
vgg = nn.Sequential(
    Normalization([0.485,0.456,0.406],[0.229,0.224,0.225],device),
    vgg
)
vgg = vgg.eval()
vgg = vgg.to(device)

modified_model = modified_vgg(vgg,idx_list,device)


attack = torchattacks.PGD(vgg,eps = 8/255)

for i,(x,y) in enumerate(test_loader):
    x = x.to(device)
    
    out_modified = modified_model(x)
    out = vgg(x)
    adv_img = attack(x,y)
    out_modified_adv = modified_model(adv_img)
    out_adv = vgg(adv_img)
    # print (out.argmax().item() == out_modified.argmax().item())
    # print (out_modified_adv.argmax().item() == out_adv.argmax().item())
    print (out.argmax().item(),out_modified.argmax().item(),out_adv.argmax().item(),out_modified_adv.argmax().item())
    print ('-------')
    
