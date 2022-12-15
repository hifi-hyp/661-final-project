import torchvision.datasets as datasets
import module
import torch.nn as nn
from imagecorruptions import corrupt
import argparse
from utils import setup_seed
import os
import torch
import torchvision.transforms as transforms
from regularization import mixup_data, Cutout, Rotation, Batch_Cutout


setup_seed(42)
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", help="index of gpu", type=int, default=0)
parser.add_argument('--alpha', default=1., type=float,
                    help='mixup interpolation coefficient (default: 1)')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    print("Run on GPU...")
else:
    print("Run on CPU...")

corruption = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur']
test_data = datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]))
test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=1, shuffle=False,
    num_workers=4, pin_memory=True)

model = module.ResNet([3, 3, 3], 10)
model.rot_head = nn.Linear(64, 4)
model.load_state_dict(torch.load('./saved_model/final_ResNet_comb.pth')['state_dict'])

model.eval()
#######################
total_examples = 0
correct_examples = 0
val_loss = 0
# disable gradient during validation, which can save GPU memory
with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        ####################################
        # copy inputs to device
        # compute the output and loss
        inputs = torch.from_numpy(corrupt(inputs.numpy(), corruption_name=corruption[batch_idx % 5], severity=1))
        inputs = Rotation(inputs)
        # print(inputs.shape)
        inputs, targets = inputs.to(device), targets.to(device)
        sz = len(targets)
        outputs, penultimate = model(inputs)
        # print(penultimate.shape)
        y_rot = torch.cat((torch.zeros(sz), torch.ones(sz), 2 * torch.ones(sz),
                           3 * torch.ones(sz)), 0).long().to(device)
        outputs = outputs[:sz]
        pred_rot = model.rot_head(penultimate[:4 * sz])
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