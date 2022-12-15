# useful libraries
# import necessary dependencies
import argparse
import os, sys
import time
import datetime
import torchvision.datasets as datasets
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torchvision
import torchvision.transforms as transforms
from regularization import mixup_data, Cutout, Rotation, Batch_Cutout
import torch.optim as optim
import module
from utils import setup_seed

setup_seed(42)
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", help="index of gpu", type=int, default=0)
parser.add_argument("--regularization", help="type of regularization, (none, cutout, mixup, rotate)", type=str,
                    default="comb")
parser.add_argument('--alpha', default=1., type=float,
                    help='mixup interpolation coefficient (default: 1)')
args = parser.parse_args()
#############################################
batch_size = 128

transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

cutout = transforms.Compose([Batch_Cutout(n_holes=1, length=16)])

train_data = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_data = datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]))


train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=batch_size, shuffle=True,
    num_workers=4, pin_memory=True)

val_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=batch_size, shuffle=False,
    num_workers=4, pin_memory=True)

# define loss function (criterion) and optimizer
criterion = nn.CrossEntropyLoss().cuda()
#############################################

net = module.ResNet([3, 3, 3], 10)
net.rot_head = nn.Linear(64, 4)

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    print("Run on GPU...")
else:
    print("Run on CPU...")


# hyperparameters, do NOT change right now
# initial learning rate

INITIAL_LR = 0.1

# momentum for optimizer
MOMENTUM = 0.9

# L2 regularization strength
REG = 1e-4


#############################################
# create loss function
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def rotate_criterion(criterion, pred, pred_rot, y, y_rot, lam):
    return criterion(pred, y) + lam * criterion(pred_rot, y_rot)


def comb_criterion(criterion, pred, pred_rot, y_a, y_b, y_rot, lam1, lam2):
    return lam1 * criterion(pred, y_a) + (1 - lam1) * criterion(pred, y_b) + lam2 * criterion(pred_rot, y_rot)


criterion = nn.CrossEntropyLoss().to(device)

# Add optimizer
optimizer = optim.SGD(net.parameters(), lr=INITIAL_LR, momentum=MOMENTUM, weight_decay=REG)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120, 160], gamma=0.1, last_epoch=-1)
# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
#############################################

# some hyperparameters
# total number of training epochs
EPOCHS = 200
net = net.to(device)
# the folder where the trained model is saved
CHECKPOINT_FOLDER = "./saved_model"

# start the training/validation process
# the process should take about 5 minutes on a GTX 1070-Ti
# if the code is written efficiently.
best_val_acc = 0
current_learning_rate = INITIAL_LR
# DECAY_EPOCHS = 10
# DECAY = 0.1


print("==> Training starts!")
print("=" * 50)
for i in range(0, EPOCHS):
    # handle the learning rate scheduler.
    # if i % DECAY_EPOCHS == 0 and i != 0:
    #     current_learning_rate = current_learning_rate * DECAY
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = current_learning_rate
    #     print("Current learning rate has decayed to %f" %current_learning_rate)

    #######################
    # switch to train mode
    net.train()
    #######################
    print("Epoch %d:" % i)
    total_examples = 0
    correct_examples = 0
    train_loss = 0
    # Train the model for 1 epoch.
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        ####################################
        # copy inputs to device
        # compute the output and loss
        inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets_a, targets_b, lam = mixup_data(inputs, targets,
                                                       args.alpha, device)
        inputs, targets_a, targets_b = map(Variable, (inputs,
                                                      targets_a, targets_b))
        inputs = cutout(inputs)
        inputs = Rotation(inputs)
        sz = len(targets)

        outputs, penultimate = net(inputs)
        y_rot = torch.cat((torch.zeros(sz), torch.ones(sz), 2 * torch.ones(sz),
                           3 * torch.ones(sz)), 0).long().to(device)
        outputs = outputs[:sz]
        pred_rot = net.rot_head(penultimate[:4 * sz])
        loss = comb_criterion(criterion, outputs, pred_rot, targets_a, targets_b, y_rot, lam, 0.5)

        # zero the gradient
        optimizer.zero_grad()
        # backpropagation
        loss.backward()
        # apply gradient and update the weights
        optimizer.step()
        # count the number of correctly predicted samples in the current batch
        train_loss += loss.item()
        _, prediction = outputs.max(1)
        total_examples += targets.size(0)
        correct_examples += lam * (prediction == targets_a.data).sum().float() + \
                            (1 - lam) * (prediction == targets_b.data).sum().float()
        ####################################
    avg_loss = train_loss / len(train_loader)
    avg_acc = correct_examples / total_examples
    print("Training loss: %.4f, Training accuracy: %.4f" % (avg_loss, avg_acc))
    # Validate on the validation dataset
    #######################
    # switch to eval mode

    # def test(net, test_loader):
    net.eval()
    #######################
    total_examples = 0
    correct_examples = 0
    val_loss = 0
    # disable gradient during validation, which can save GPU memory
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            ####################################
            # copy inputs to device
            # compute the output and loss
            inputs = Rotation(inputs)
            # print(inputs.shape)
            inputs, targets = inputs.to(device), targets.to(device)
            sz = len(targets)
            outputs, penultimate = net(inputs)
            # print(penultimate.shape)
            y_rot = torch.cat((torch.zeros(sz), torch.ones(sz), 2 * torch.ones(sz),
                               3 * torch.ones(sz)), 0).long().to(device)
            outputs = outputs[:sz]
            pred_rot = net.rot_head(penultimate[:4 * sz])
            # def rotate_criterion(criterion, pred, rot_pred, y, y_rot, lam):
            loss = rotate_criterion(criterion, outputs, pred_rot, targets, y_rot, 0.5)
            # count the number of correctly predicted samples in the current batch
            val_loss += loss.item()
            _, prediction = outputs.max(1)
            total_examples += targets.size(0)
            correct_examples += (prediction == targets).sum().item()
            ####################################

    avg_loss = val_loss / len(val_loader)
    avg_acc = correct_examples / total_examples
    print("Validation loss: %.4f, Validation accuracy: %.4f" % (avg_loss, avg_acc))

    scheduler.step()

    # # save the model checkpoint
    if avg_acc > best_val_acc:
        best_val_acc = avg_acc
        if not os.path.exists(CHECKPOINT_FOLDER):
            os.makedirs(CHECKPOINT_FOLDER)
        print("Saving ...")
        state = {'state_dict': net.state_dict(),
                 'epoch': i,
                 'lr': current_learning_rate}
        torch.save(state, os.path.join(CHECKPOINT_FOLDER, 'best_ResNet_' + args.regularization + '.pth'))

    print('')

print("Final Saving ...")
state = {'state_dict': net.state_dict(),
         'epoch': i,
         'lr': current_learning_rate}
torch.save(state, os.path.join(CHECKPOINT_FOLDER, 'final_ResNet_' + args.regularization + '.pth'))


print("=" * 50)
print(f"==> Optimization finished! Best validation accuracy: {best_val_acc:.4f}")
