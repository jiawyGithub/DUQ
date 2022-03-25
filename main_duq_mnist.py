import torch
import torch.utils.data
from torchvision import datasets
from torch.nn import functional as F
import torchvision.transforms as transforms

from model.cnn_duq import CNN_DUQ
from config.train_config import get_args
from utils.datasets import MNIST

use_gpu = torch.cuda.is_available()
args = get_args()

def accuracy(pred, target, topk=(1,)):
#     correct = pred.eq(target.view(1, -1).expand_as(pred))
#     res = []
#     for k in topk:
#         correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
#         res.append(correct_k.mul_(100.0 / batch_size))
#     return res
    return 1
def bce():
    return 1

def gradient_penalty():
    return 1

def train_model(model, optimizer, train_loader):
    model.train()
    correct = 0
    
    for i, (x, target) in enumerate(train_loader):
        y = F.one_hot(target, num_classes=10).float()
        if use_gpu:
            x, y = x.cuda(), y.cuda()

        x.requires_grad_(True)
        z, y_pred = model(x)
        loss = F.binary_cross_entropy(y_pred, y)
        x.requires_grad_(False)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            model.update_embeddings(x, y)
               
        _, _target_pred = y_pred.topk(1)
        target_pred = _target_pred.view_as(target)
        correct += target.eq(target_pred).sum().item()
        print('correct', correct)

    percentage_correct = 100.0 * correct / len(train_loader.dataset)
    return percentage_correct

def main():
    
    print('loading dataset...')
    train_dataset = MNIST(transform=transforms.ToTensor(), root="data/", train=True, download=True) 
    test_dataset = MNIST(transform=transforms.ToTensor(),root="data/", train=False, download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=128,
                                               num_workers=0,
                                               drop_last=True,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=2000, 
                                              num_workers=0,
                                              shuffle=False)

    print('building model...')
    input_size = 28
    num_classes = 10
    embedding_size = 256
    learnable_length_scale = False
    gamma = 0.999
    length_scale = 0.05
    model = CNN_DUQ(input_size,num_classes,embedding_size,learnable_length_scale,length_scale,gamma,)
    if use_gpu:
        model = model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)

    for epoch in range(1, args.n_epoch+1):  
        Acc = train_model(model, optimizer, train_loader)
        # Acc = accuracy()
        # BCE = bce()
        # GP = gradient_penalty()

        print(
            f"Validation Results - Epoch: {epoch} "
            f"Acc: {Acc:.4f}%"
            # f"BCE: {BCE:.2f} "
            # f"GP: {GP:.6f} "
        )
        print(f"Sigma: {model.sigma}")

if __name__=='__main__':
    main()