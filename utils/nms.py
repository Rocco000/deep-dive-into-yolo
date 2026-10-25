import torch
from utils.box_formats import BoxFormat
from utils.boxes import middlepoint_to_cornerpoint
from utils.metrics import intersection_over_union

def non_max_suppression(bboxes: list[torch.Tensor], iou_threshold: float = 0.5, score_threshold: float = 0.5, box_format: BoxFormat = BoxFormat.CORNER_POINT) -> list[torch.Tensor]:
    """
    Perform non-maximum suppression (NMS) on the given bounding boxes.
    Cleans up overlapping boxes based on their confidence scores and a specified IoU threshold.

    Args:
        bboxes (list[torch.Tensor]): A list of tensors, each containing bounding boxes in the format (class, score, x1, y1, x2, y2).
        iou_threshold (float): The Intersection over Union (IoU) threshold for suppressing overlapping boxes. Default is 0.5.
        score_threshold (float): The confidence score threshold for filtering boxes. Default is 0.5.
        box_format (BoxFormat): The format of the bounding boxes. Default is BoxFormat.CORNER_POINT.

    Returns:
        list[torch.Tensor]: A list of tensors, each containing the remaining bounding boxes after NMS.
    """
    if not isinstance(bboxes, list) or not all(isinstance(box, torch.Tensor) for box in bboxes):
        raise TypeError(f"bboxes must be a list of torch.Tensor, got {type(bboxes)}")
    if not isinstance(iou_threshold, float):
        raise TypeError(f"iou_threshold must be a float, got {type(iou_threshold)}")
    if not isinstance(score_threshold, float):
        raise TypeError(f"score_threshold must be a float, got {type(score_threshold)}")
    if not isinstance(box_format, BoxFormat):
        raise TypeError(f"box_format must be an instance of BoxFormat, got {type(box_format)}")

    if not (0 <= iou_threshold <= 1):
        raise ValueError(f"iou_threshold must be a float between 0 and 1, got {iou_threshold}")
    if not (0 <= score_threshold <= 1):
        raise ValueError(f"score_threshold must be a float between 0 and 1, got {score_threshold}")

    if not all(box.ndim == 1 and box.shape[0] == 6 for box in bboxes):
        raise ValueError("Each bounding box tensor must be of shape (6), where 6 corresponds to (class, score, x1, y1, x2, y2).")

    # Filter boxes based on score threshold
    filtered_bboxes: list[torch.Tensor] = [box for box in bboxes if box[1] > score_threshold]

    # Sort the filtered boxes by score in descending order
    filtered_bboxes = sorted(filtered_bboxes, key=lambda x: x[1], reverse=True)

    # Perform NMS for each class separately
    final_bboxes: list[torch.Tensor] = []
    while filtered_bboxes:
        chosen_box: torch.Tensor = filtered_bboxes.pop(0)

        # Keep boxes that have different class or have IoU less than the threshold with the chosen box
        filtered_bboxes = [
            box
            for box in filtered_bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(chosen_box[2:].unsqueeze(0), box[2:].unsqueeze(0), box_format) < iou_threshold
        ]

        final_bboxes.append(chosen_box)

    return final_bboxes