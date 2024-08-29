
import torch
import torch.nn as nn
import torch.nn.functional as F
class MultiClassInstanceSegmentationLoss(nn.Module):
    def __init__(self, weight_ce=1.0, weight_dice=1.0, weight_iou=1.0, num_classes=3):
        super(MultiClassInstanceSegmentationLoss, self).__init__()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.weight_iou = weight_iou
        self.num_classes = num_classes
        self.ce_loss = nn.CrossEntropyLoss()

    def dice_loss(self, outputs, labels, smooth=1e-6):
        dice = 0
        for c in range(self.num_classes):
            outputs_c = outputs[:, c]
            labels_c = (labels == c).float()
            intersection = (outputs_c * labels_c).sum(dim=(1, 2))
            union = outputs_c.sum(dim=(1, 2)) + labels_c.sum(dim=(1, 2))
            dice += (2. * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()

    def iou_loss(self, outputs, labels, smooth=1e-6):
        iou = 0
        for c in range(self.num_classes):
            outputs_c = outputs[:, c]
            labels_c = (labels == c).float()
            intersection = (outputs_c * labels_c).sum(dim=(1, 2))
            union = outputs_c.sum(dim=(1, 2)) + labels_c.sum(dim=(1, 2)) - intersection
            iou += (intersection + smooth) / (union + smooth)
        return 1 - iou.mean()

    def forward(self, outputs, labels):
        ce_loss = self.ce_loss(outputs, labels)

        # Apply softmax to outputs for multi-class classification
        outputs = F.softmax(outputs, dim=1)

        dice_loss = self.dice_loss(outputs, labels)
        iou_loss = self.iou_loss(outputs, labels)

        total_loss = (self.weight_ce * ce_loss) + (self.weight_dice * dice_loss) + (self.weight_iou * iou_loss)
        return total_loss


# AI GENERATED
def iou_score(pred_mask, true_mask, threshold=0.5, smooth=1e-6):
    """
    Compute the IoU score between the predicted mask and the true mask for multiple channels.

    Parameters:
    - pred_mask: Predicted mask (tensor) of shape (N, C, H, W), where N is the batch size, C is the number of channels, H is the height, and W is the width.
    - true_mask: Ground truth mask (tensor) of shape (N, C, H, W), where N is the batch size, C is the number of channels, H is the height, and W is the width.
    - threshold: Threshold to binarize the predicted mask. Default is 0.5.
    - smooth: Smoothing factor to avoid division by zero. Default is 1e-6.

    Returns:
    - IoU score (tensor) of shape (N, C) for each mask in the batch across all channels.
    """

    # Apply threshold to predicted mask to binarize it
    pred_mask = (pred_mask > threshold).float()

    # Flatten the masks along H and W
    pred_mask = pred_mask.view(pred_mask.size(0), pred_mask.size(1), -1)  # (N, C, H*W)
    true_mask = true_mask.view(true_mask.size(0), true_mask.size(1), -1)  # (N, C, H*W)

    # Compute intersection and union per channel
    intersection = (pred_mask * true_mask).sum(dim=2)  # Sum over H*W
    union = pred_mask.sum(dim=2) + true_mask.sum(dim=2) - intersection  # Sum over H*W

    # Compute IoU per channel
    iou = (intersection + smooth) / (union + smooth)  # Shape (N, C)

    return iou