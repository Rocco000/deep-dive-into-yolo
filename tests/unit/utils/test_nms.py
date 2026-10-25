import torch
import pytest

from utils.nms import non_max_suppression as nms
from utils.box_formats import BoxFormat

@pytest.mark.unit
class TestNonMaxSuppression:

    @pytest.mark.boundary
    def test_wrong_tensor_type(self):
        with pytest.raises(TypeError):
            nms(
                bboxes=[1, 2, 3],
                iou_threshold=0.5,
                score_threshold=0.5,
                box_format=BoxFormat.CORNER_POINT,
            )

    @pytest.mark.boundary
    def test_not_list_of_tensors(self):
        with pytest.raises(TypeError):
            nms(
                bboxes="not-a-list",
                iou_threshold=0.5,
                score_threshold=0.5,
                box_format=BoxFormat.CORNER_POINT,
            )

    @pytest.mark.boundary
    def test_wrong_iou_threshold_type(self):
        with pytest.raises(TypeError):
            nms(
                bboxes=[torch.tensor([1, 2, 3, 4, 5, 6])],
                iou_threshold="not-a-float",
                score_threshold=0.5,
                box_format=BoxFormat.CORNER_POINT,
            )

    @pytest.mark.boundary
    def test_wrong_score_threshold_type(self):
        with pytest.raises(TypeError):
            nms(
                bboxes=[torch.tensor([1, 2, 3, 4, 5, 6])],
                iou_threshold=0.5,
                score_threshold="not-a-float",
                box_format=BoxFormat.CORNER_POINT,
            )

    @pytest.mark.boundary
    def test_wrong_box_format_type(self):
        with pytest.raises(TypeError):
            nms(
                bboxes=[torch.tensor([1, 2, 3, 4, 5, 6])],
                iou_threshold=0.5,
                score_threshold=0.5,
                box_format="not-a-box-format",
            )

    @pytest.mark.boundary
    def test_wrong_iou_threshold_value(self):
        with pytest.raises(ValueError):
            nms(
                bboxes=[torch.tensor([1, 2, 3, 4, 5, 6])],
                iou_threshold=1.5,
                score_threshold=0.5,
                box_format=BoxFormat.CORNER_POINT,
            )

    @pytest.mark.boundary
    def test_wrong_score_threshold_value(self):
        with pytest.raises(ValueError):
            nms(
                bboxes=[torch.tensor([1, 2, 3, 4, 5, 6])],
                iou_threshold=0.5,
                score_threshold=-0.1,
                box_format=BoxFormat.CORNER_POINT,
            )

    @pytest.mark.boundary
    def test_wrong_bbox_shape(self):
        with pytest.raises(ValueError):
            nms(
                bboxes=[torch.tensor([1, 2])],
                iou_threshold=0.5,
                score_threshold=0.5,
                box_format=BoxFormat.CORNER_POINT,
            )

    def test_remove_on_iou(self):
        t1_boxes = [
            torch.tensor([1, 1, 0.5, 0.45, 0.4, 0.5]),
            torch.tensor([1, 0.8, 0.5, 0.5, 0.2, 0.4]),
            torch.tensor([1, 0.7, 0.25, 0.35, 0.3, 0.1]),
            torch.tensor([1, 0.05, 0.1, 0.1, 0.1, 0.1]),
        ]

        expected_boxes = [torch.tensor([1, 1, 0.5, 0.45, 0.4, 0.5]), torch.tensor([1, 0.7, 0.25, 0.35, 0.3, 0.1])]
        expected_boxes = sorted(expected_boxes, key=lambda x: x[1], reverse=True)

        bboxes = nms(
            t1_boxes,
            iou_threshold=7 / 20,
            score_threshold=0.2,
            box_format=BoxFormat.MIDDLE_POINT,
        )
        bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)

        assert len(bboxes) == len(expected_boxes)
        assert all(torch.equal(box, expected_box) for box, expected_box in zip(bboxes, expected_boxes))

    def test_keep_on_class(self):
        t2_boxes = [
            torch.tensor([1, 1, 0.5, 0.45, 0.4, 0.5]),
            torch.tensor([2, 0.9, 0.5, 0.5, 0.2, 0.4]),
            torch.tensor([1, 0.8, 0.25, 0.35, 0.3, 0.1]),
            torch.tensor([1, 0.05, 0.1, 0.1, 0.1, 0.1]),
        ]

        expected_boxes = [
            torch.tensor([1, 1, 0.5, 0.45, 0.4, 0.5]),
            torch.tensor([2, 0.9, 0.5, 0.5, 0.2, 0.4]),
            torch.tensor([1, 0.8, 0.25, 0.35, 0.3, 0.1]),
        ]
        expected_boxes = sorted(expected_boxes, key=lambda x: x[1], reverse=True)

        bboxes = nms(
            t2_boxes,
            iou_threshold=7 / 20,
            score_threshold=0.2,
            box_format=BoxFormat.MIDDLE_POINT,
        )
        bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)

        assert len(bboxes) == len(expected_boxes)
        assert all(torch.equal(box, expected_box) for box, expected_box in zip(bboxes, expected_boxes))

    def test_remove_on_iou_and_class(self):
        t3_boxes = [
            torch.tensor([1, 0.9, 0.5, 0.45, 0.4, 0.5]),
            torch.tensor([1, 1, 0.5, 0.5, 0.2, 0.4]),
            torch.tensor([2, 0.8, 0.25, 0.35, 0.3, 0.1]),
            torch.tensor([1, 0.05, 0.1, 0.1, 0.1, 0.1]),
        ]

        expected_boxes = [torch.tensor([1, 1, 0.5, 0.5, 0.2, 0.4]), torch.tensor([2, 0.8, 0.25, 0.35, 0.3, 0.1])]
        expected_boxes = sorted(expected_boxes, key=lambda x: x[1], reverse=True)

        bboxes = nms(
            t3_boxes,
            iou_threshold=7 / 20,
            score_threshold=0.2,
            box_format=BoxFormat.MIDDLE_POINT,
        )
        bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)

        assert len(bboxes) == len(expected_boxes)
        assert all(torch.equal(box, expected_box) for box, expected_box in zip(bboxes, expected_boxes))

    def test_keep_on_iou(self):
        t4_boxes = [
            torch.tensor([1, 0.9, 0.5, 0.45, 0.4, 0.5]),
            torch.tensor([1, 1, 0.5, 0.5, 0.2, 0.4]),
            torch.tensor([1, 0.8, 0.25, 0.35, 0.3, 0.1]),
            torch.tensor([1, 0.05, 0.1, 0.1, 0.1, 0.1]),
        ]

        expected_boxes = [
            torch.tensor([1, 0.9, 0.5, 0.45, 0.4, 0.5]),
            torch.tensor([1, 1, 0.5, 0.5, 0.2, 0.4]),
            torch.tensor([1, 0.8, 0.25, 0.35, 0.3, 0.1]),
        ]
        expected_boxes = sorted(expected_boxes, key=lambda x: x[1], reverse=True)

        bboxes = nms(
            t4_boxes,
            iou_threshold=9 / 20,
            score_threshold=0.2,
            box_format=BoxFormat.MIDDLE_POINT,
        )
        bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)

        assert len(bboxes) == len(expected_boxes)
        assert all(torch.equal(box, expected_box) for box, expected_box in zip(bboxes, expected_boxes))