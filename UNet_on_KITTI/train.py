from UNet_on_KITTI.dice_loss import dice_loss

def train(model, data_l, learning_rate):
    cost = []
    for img, msk in data_l:
        optimizer.zero_grad()
        out = model(img.to(device))
        loss = DiceLoss(out, msk.to(device))
        loss.backward()
        optimizer.step()
        cost.append(loss.item())

    return cost