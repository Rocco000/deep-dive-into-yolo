import torch
from utils.box_formats import BoxFormat
from utils.boxes import middlepoint_to_cornerpoint

def intersection_over_union(predicted_boxes: torch.Tensor, labels_boxes: torch.Tensor, box_format: BoxFormat=BoxFormat.MIDDLE_POINT) -> torch.Tensor:
    """
    Compute the intersection over union (IoU) between predicted bounding boxes and ground truth bounding boxes.

    Args:
        predicted_boxes (torch.Tensor): A tensor of shape (N, 4) containing the predicted bounding boxes in the format (x1, y1, x2, y2).
        labels_boxes (torch.Tensor): A tensor of shape (M, 4) containing the ground truth bounding boxes in the format (x1, y1, x2, y2).
        box_format (BoxFormat): The format of the bounding boxes. Default is BoxFormat.MIDDLE_POINT.

    Returns:
        torch.Tensor: The intersection over union (IoU) between the predicted and ground truth bounding boxes. It is a tensor of shape (N, ) containing the IoU values for each pair of predicted and ground truth boxes.
    """
    if not isinstance(predicted_boxes, torch.Tensor):
        raise TypeError(f"predicted_boxes must be a torch.Tensor, got {type(predicted_boxes)}")
    if not isinstance(labels_boxes, torch.Tensor):
        raise TypeError(f"labels_boxes must be a torch.Tensor, got {type(labels_boxes)}")
    if not isinstance(box_format, BoxFormat):
        raise TypeError(f"box_format must be an instance of BoxFormat, got {type(box_format)}")

    if predicted_boxes.ndim != 2 or predicted_boxes.shape[1] != 4:
        raise ValueError(f"predicted_boxes must be of shape (N, 4), got {predicted_boxes.shape}")
    if labels_boxes.ndim != 2 or labels_boxes.shape[1] != 4:
        raise ValueError(f"labels_boxes must be of shape (M, 4), got {labels_boxes.shape}")

    if predicted_boxes.shape[0] != labels_boxes.shape[0]:
        raise ValueError(f"predicted_boxes and labels_boxes must have the same number of boxes, got predicted: {predicted_boxes.shape[0]} and labels: {labels_boxes.shape[0]}")

    if box_format == BoxFormat.MIDDLE_POINT:
        # Convert middle point format to corner point format
        predicted_boxes = middlepoint_to_cornerpoint(predicted_boxes)
        labels_boxes = middlepoint_to_cornerpoint(labels_boxes)

    # Get box coordinates
    pred_x1 = predicted_boxes[..., 0:1] # (N, 1) / (batch_size, 1)
    pred_y1 = predicted_boxes[..., 1:2] # (N, 1)
    pred_x2 = predicted_boxes[..., 2:3] # (N, 1)
    pred_y2 = predicted_boxes[..., 3:4] # (N, 1)

    label_x1 = labels_boxes[..., 0:1] # (M, 1)
    label_y1 = labels_boxes[..., 1:2] # (M, 1)
    label_x2 = labels_boxes[..., 2:3] # (M, 1)
    label_y2 = labels_boxes[..., 3:4] # (M, 1)

    # Compute intersection area
    inter_x1 = torch.max(pred_x1, label_x1)
    inter_y1 = torch.max(pred_y1, label_y1)
    inter_x2 = torch.min(pred_x2, label_x2)
    inter_y2 = torch.min(pred_y2, label_y2)

    # clamp(min=0) to handle cases where there is no intersection
    inter_width = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_height = torch.clamp(inter_y2 - inter_y1, min=0)

    inter_area = inter_width * inter_height

    # Compute union area
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    label_area = (label_x2 - label_x1) * (label_y2 - label_y1)

    union_area = pred_area + label_area - inter_area + 1e-6 # Add a small epsilon to  have training stability (avoid division by zero)

    return inter_area / union_area
    