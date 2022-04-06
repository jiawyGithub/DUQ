import torch
import torch.utils.data
from torch.nn import functional as F
from torchvision import datasets, transforms

import json

from model.cnn_duq import CNN_DUQ
from config.deq_minist_config import get_args

transform = transforms.ToTensor()

use_gpu = torch.cuda.is_available()
args = get_args()
kwargs = vars(args)
print("input args:\n", json.dumps(kwargs, indent=4, separators=(",", ":")))

train_data = {
    "mnist": datasets.MNIST(root="data/", train=True, download=True, transform=transform),
    "fashion_mnist": datasets.FashionMNIST(root="data/", train=True, download=True, transform=transform)
}
test_data = {
    "mnist": datasets.MNIST(root="data/", train=False, download=True, transform=transform),
    "fashion_mnist": datasets.FashionMNIST(root="data/", train=False, download=True, transform=transform)
}

def train_model(model, optimizer, train_loader):
    model.train()
    total_correct = 0
    total_loss = 0
    
    for i, (x, target) in enumerate(train_loader):
        y = F.one_hot(target, num_classes=10).float()
        if use_gpu:
            x, y, target = x.cuda(), y.cuda(), target.cuda()

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

        # 计算预测正确的样本个数和总损失
        _, target_pred = y_pred.topk(1)
        correct = target.eq(target_pred.view_as(target)).sum().item()
        total_correct += correct
        total_loss += loss.item()
        if args.debug and  (i+1) % 10 == 0:
            print("Iter", i+1, "correct", correct, "loss", loss.item())

    # 计算准确率和平均loss
    l = len(train_loader.dataset)
    percentage_correct = 100.0 * total_correct / l
    avg_loss = total_loss / l

    return percentage_correct, avg_loss

def evaluate(model, test_loader):
    model.eval()
    correct = 0

    for i, (x, target) in enumerate(test_loader):

        y = F.one_hot(target, num_classes=10).float()
        if use_gpu:
            x, y, target= x.cuda(), y.cuda(), target.cuda()
        x.requires_grad_(True)

        z, y_pred = model(x)

        # 计算预测正确的样本个数
        _, _target_pred = y_pred.topk(1)
        target_pred = _target_pred.view_as(target)
        correct += target.eq(target_pred).sum().item()
    
    # 计算准确率和平均loss
    l = len(test_loader.dataset)
    percentage_correct = 100.0 * correct / l

    return percentage_correct

def main():
    print(f'loading dataset <{args.dataset}>...')
    train_dataset = train_data[args.dataset] 
    test_dataset = test_data[args.dataset]
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.num_workers,
                                               drop_last=True,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.batch_size, 
                                              num_workers=args.num_workers,
                                              shuffle=False)

    print('building model...')
    input_size = 28
    num_classes = 10
    embedding_size = 256
    gamma = 0.999
    learnable_length_scale = False
    length_scale = 0.05
    model = CNN_DUQ(input_size,num_classes,embedding_size,learnable_length_scale,length_scale,gamma,)
    if use_gpu:
        model = model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentun, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.scheduler_gama) # 调整学习率

    print('training...')
    for epoch in range(1, args.n_epoch+1):  
        acc, avg_loss = train_model(model, optimizer, train_loader)
        scheduler.step()
        
        print(
            f"Validation Results - Epoch: {epoch} "
            f"Acc: {acc:.4f}% "
            f"AvgLoss: {avg_loss:.4f} "
        )
        print(f"Sigma: {model.sigma}")

        test_acc = evaluate(model, test_loader)
        print(
            f"TestAcc: {test_acc:.4f}% "
        )

if __name__=='__main__':
    main()