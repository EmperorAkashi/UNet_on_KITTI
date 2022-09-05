from UNet_on_KITTI.dice_loss import dice_loss


from UNet_on_KITTI.dice_loss import dice_loss
import torch

def eval_loop(model, optimizer, eval_l):  #evaluate the test set
    running_loss = 0
    model.eval()
    with torch.no_grad():
        f1_scores, accuracy = [], []
        for imgs, masks in eval_l:
            # pass to device
            imgs = imgs.to(device)
            masks = masks.to(device)
            # forward
            out = model(imgs)
            loss = dice_loss(out, masks)
            running_loss += loss.item()*imgs.shape[0]
            # calculate predictions using output
            predicted = (out > 0.5).float()  
            predicted = predicted.view(-1).cpu().numpy()   #turn tensor to array for accuracy
            masks = (masks > 0.5).float()  
            labels = masks.view(-1).cpu().numpy()
            accuracy.append(accuracy_score(labels, predicted))
            f1_scores.append(f1_score(labels, predicted))
    acc = sum(accuracy)/len(accuracy)
    f1 = sum(f1_scores)/len(f1_scores)
    running_loss /= len(img_test)
    return acc, f1, running_loss