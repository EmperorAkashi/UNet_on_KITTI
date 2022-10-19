import argparse
from UNet_on_KITTI.dice_loss import dice_loss
from UNet_on_KITTI.neuron_dataloader import Data
from UNet_on_KITTI.model import UNet

def train_step(model, data_l, learning_rate):
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    device = torch.device('cpu')

    cost = []
    for img, msk in data_l:
        optimizer.zero_grad()
        out = model(img.to(device))
        loss = dice_loss(out, msk.to(device))
        loss.backward()
        optimizer.step()
        cost.append(loss.item())

    return cost

def train(args):
    cost_list = []
    for e in range(num_epoch):
        mini_cost = train(model, data_l, learning_rate)
        cost_list += mini_cost
    return cost_list
    

parser = argparse.ArgumentParser(description="segmentation training")
parser.add_argument('--model', default = UNet)
parser.add_argument('--train_dir', default = '../input/sartorius-cell-instance-segmentation', help = 'train image directory')
parser.add_argument('--num_classes', default = 1)
parser.add_argument('--num_epoch', default = 10)
parser.add_argument('--learning_rate', default = 0.01)

if __name__ == "__main__":
    args = parser.parse_args()
    train(args)