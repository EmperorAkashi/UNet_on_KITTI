import argparse
from UNet_on_KITTI.dice_loss import dice_loss

def train(model, data_l, learning_rate):
    cost = []
    for img, msk in data_l:
        optimizer.zero_grad()
        out = model(img.to(device))
        loss = dice_loss(out, msk.to(device))
        loss.backward()
        optimizer.step()
        cost.append(loss.item())

    return cost

parser = argparse.ArgumentParser(description="segmentation training")

if __name__ == "__main__":
    args = parser.parse_args()
    train(args)