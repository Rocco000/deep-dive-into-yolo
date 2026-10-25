import torch

def middlepoint_to_cornerpoint(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert bounding boxes from middle point format (x_center, y_center, width, height) to corner point format (x1, y1, x2, y2).

    Args:
        boxes (torch.Tensor): A tensor of shape (N, 4) containing the bounding boxes in middle point format.

    Returns:
        torch.Tensor: A tensor of shape (N, 4) containing the bounding boxes in corner point format.
    """
    if not isinstance(boxes, torch.Tensor):
        raise TypeError(f"boxes must be a torch.Tensor, got {type(boxes)}")

    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"boxes must be of shape (N, 4), got {boxes.shape}")

    x_center = boxes[..., 0:1] # (N, 1) / (batch_size, 1)
    y_center = boxes[..., 1:2] # (N, 1)
    width = boxes[..., 2:3] # (N, 1)
    height = boxes[..., 3:4] # (N, 1)

    x1 = x_center - (width / 2)
    y1 = y_center - (height / 2)
    x2 = x_center + (width / 2)
    y2 = y_center + (height / 2)

    return torch.cat([x1, y1, x2, y2], dim=-1)