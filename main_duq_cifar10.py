import torch
import torch.utils.data
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.models import resnet50

import json

from model.resnet_duq import ResNet_DUQ
from config.deq_cifar10_config import get_args

transform = transforms.ToTensor()

use_gpu = torch.cuda.is_available()
args = get_args()
kwargs = vars(args)
print("input args:\n", json.dumps(kwargs, indent=4, separators=(",", ":")))

train_data = {
    "cifar10": datasets.CIFAR10(root="data/CIFAR10/", train=True, download=True, transform=transform),
}
test_data = {
    "cifar10": datasets.CIFAR10(root="data/CIFAR10/", train=False, download=True, transform=transform),
}

def calc_gradients_input(x, y_pred):
    gradients = torch.autograd.grad(
        outputs=y_pred,
        inputs=x,
        grad_outputs=torch.ones_like(y_pred),
        create_graph=True,
    )[0] # Size([128, 3, 32, 32])

    gradients = gradients.flatten(start_dim=1)  # [128, 3072]

    return gradients

def calc_gradient_penalty(x, y_pred):
    # 计算梯度
    gradients = calc_gradients_input(x, y_pred)

    # L2 norm 2范数
    grad_norm = gradients.norm(2, dim=1) 

    # Two sided penalty
    gradient_penalty = ((grad_norm - 1) ** 2).mean()

    return gradient_penalty

def train_model(model, optimizer, train_loader):
    model.train()
    total_correct = 0
    total_loss = 0
    total_bce = 0
    total_gp = 0
    l_gradient_penalty = 0.75
    
    for i, (x, target) in enumerate(train_loader):
        y = F.one_hot(target, num_classes=10).float()
        if use_gpu:
            x, y, target = x.cuda(), y.cuda(), target.cuda()

        x.requires_grad_(True)
        y_pred = model(x)
        bce = F.binary_cross_entropy(y_pred, y)
        loss = bce
        if l_gradient_penalty > 0: # 0.75 two-sided penalty 就是lamda
            gp = calc_gradient_penalty(x, y_pred)
            loss += l_gradient_penalty * gp 
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
        total_bce += bce.item()
        total_gp += gp.item()

        if args.debug and  (i+1) % 2 == 0:
            print(
                f"Validation Results - Iter: {i+1} "
                f"correct: {correct} "
                f"loss: {loss.item():.4f} "
                f"BCE: {bce.item():.2f} "
                f"GP: {gp.item():.2f} "
            )

    # 计算准确率和平均loss
    l = len(train_loader.dataset)
    percentage_correct = 100.0 * total_correct / l
    avg_loss = total_loss / l
    avg_bce = total_bce / l
    avg_gp = total_gp / l

    return percentage_correct, avg_loss, avg_bce, avg_gp

def evaluate(model, test_loader):
    model.eval()
    correct = 0

    for i, (x, target) in enumerate(test_loader):

        y = F.one_hot(target, num_classes=10).float()
        if use_gpu:
            x, y, target= x.cuda(), y.cuda(), target.cuda()
        x.requires_grad_(True)

        y_pred = model(x)

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
                                               batch_size=128,
                                               num_workers=0,
                                               drop_last=True,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=128, 
                                              num_workers=0,
                                              shuffle=False)

    print('building model...')
    feature_extractor = resnet50()
    # Adapted resnet from:
    # https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
    feature_extractor.conv1 = torch.nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    feature_extractor.maxpool = torch.nn.Identity()
    feature_extractor.fc = torch.nn.Identity()

    num_classes = 10
    model_output_size = 512
    centroid_size = model_output_size
    length_scale = 0.1
    gamma = 0.999

    model = ResNet_DUQ(
        feature_extractor,
        num_classes,
        centroid_size,
        model_output_size,
        length_scale, # 0.1 超参数sigma
        gamma, # 0.999 更新质心的超参数
    )
    if use_gpu:
        model = model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.scheduler_gama)

    print('training...')
    for epoch in range(1, args.n_epoch+1):  
        acc, avg_loss, bce, gp = train_model(model, optimizer, train_loader)
        scheduler.step()
        print(
            f"Validation Results - Epoch: {epoch} "
            f"Acc: {acc:.4f}% "
            f"AvgLoss: {avg_loss:.4f} "
            f"AvgBCE: {bce:.2f} "
            f"AvgGP: {gp:.2f} "
        )
        print(f"Sigma: {model.sigma}")

        test_acc = evaluate(model, test_loader)
        print(
            f"TestAcc: {test_acc:.4f}% "
        )

if __name__=='__main__':
    main()