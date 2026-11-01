import pytest
import torch

from utils.box_formats import BoxFormat
from utils.metrics import intersection_over_union, mean_average_precision, non_max_suppression as nms

@pytest.mark.unit
class TestIntersectionOverUnion():
    # Accept if the difference in iou is smal
    epsilon = 0.001

    @pytest.mark.boundary
    def test_invalid_tensor_type(self):
        with pytest.raises(TypeError):
            intersection_over_union("not-a-tensor", torch.tensor([0, 0, 1, 1]), box_format=BoxFormat.CORNER_POINT)

    @pytest.mark.boundary
    def test_invalid_box_format_type(self):
        with pytest.raises(TypeError):
            intersection_over_union(torch.tensor([0, 0, 1, 1]), torch.tensor([0, 0, 1, 1]), box_format="not-a-box-format")

    @pytest.mark.boundary
    def test_invalid_num_dimensions(self):
        with pytest.raises(ValueError):
            intersection_over_union(torch.tensor([0, 0, 1]), torch.tensor([0, 0, 1, 1]), box_format=BoxFormat.CORNER_POINT)

    @pytest.mark.boundary
    def test_invalid_tensor_shape(self):
        with pytest.raises(ValueError):
            intersection_over_union(torch.tensor([[0, 0, 1, 1], [0, 0, 1]]), torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]]), box_format=BoxFormat.CORNER_POINT)

    def test_both_inside_cell_shares_area(self):
        t1_box1 = torch.tensor([[0.8, 0.1, 0.2, 0.2]])
        t1_box2 = torch.tensor([[0.9, 0.2, 0.2, 0.2]])
        t1_correct_iou = 1 / 7

        iou = intersection_over_union(t1_box1, t1_box2, box_format=BoxFormat.MIDDLE_POINT)
        assert (torch.abs(iou - t1_correct_iou) < self.epsilon)

    def test_partially_outside_cell_shares_area(self):
        t2_box1 = torch.tensor([[0.95, 0.6, 0.5, 0.2]])
        t2_box2 = torch.tensor([[0.95, 0.7, 0.3, 0.2]])
        t2_correct_iou = 3 / 13
        
        iou = intersection_over_union(t2_box1, t2_box2, box_format=BoxFormat.MIDDLE_POINT)
        assert (torch.abs(iou - t2_correct_iou) < self.epsilon)

    def test_both_inside_cell_shares_no_area(self):
        t3_box1 = torch.tensor([[0.25, 0.15, 0.3, 0.1]])
        t3_box2 = torch.tensor([[0.25, 0.35, 0.3, 0.1]])
        t3_correct_iou = 0
        
        iou = intersection_over_union(t3_box1, t3_box2, box_format=BoxFormat.MIDDLE_POINT)
        assert (torch.abs(iou - t3_correct_iou) < self.epsilon)

    def test_MIDDLE_POINT_outside_cell_shares_area(self):
        t4_box1 = torch.tensor([[0.7, 0.95, 0.6, 0.1]])
        t4_box2 = torch.tensor([[0.5, 1.15, 0.4, 0.7]])
        t4_correct_iou = 3 / 31
        
        iou = intersection_over_union(t4_box1, t4_box2, box_format=BoxFormat.MIDDLE_POINT)
        assert (torch.abs(iou - t4_correct_iou) < self.epsilon)

    def test_both_inside_cell_shares_entire_area(self):
        t5_box1 = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
        t5_box2 = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
        t5_correct_iou = 1

        iou = intersection_over_union(t5_box1, t5_box2, box_format=BoxFormat.MIDDLE_POINT)
        assert (torch.abs(iou - t5_correct_iou) < self.epsilon)

    def test_box_format_x1_y1_x2_y2(self):
        t6_box1 = torch.tensor([[2, 2, 6, 6]])
        t6_box2 = torch.tensor([[4, 4, 7, 8]])
        t6_correct_iou = 4 / 24

        iou = intersection_over_union(t6_box1, t6_box2, box_format=BoxFormat.CORNER_POINT)
        assert torch.abs(iou - t6_correct_iou) < self.epsilon

    def test_additional_and_batch(self):
        t12_bboxes1 = torch.tensor(
            [
                [0, 0, 2, 2],
                [0, 0, 2, 2],
                [0, 0, 2, 2],
                [0, 0, 2, 2],
                [0, 0, 2, 2],
                [0, 0, 3, 2],
            ]
        )
        t12_bboxes2 = torch.tensor(
            [
                [3, 0, 5, 2],
                [3, 0, 5, 2],
                [0, 3, 2, 5],
                [2, 0, 5, 2],
                [1, 1, 3, 3],
                [1, 1, 3, 3],
            ]
        )
        t12_correct_ious = torch.tensor([0, 0, 0, 0, 1 / 7, 0.25])

        ious = intersection_over_union(
            t12_bboxes1, t12_bboxes2, box_format=BoxFormat.CORNER_POINT
        )
        all_true = torch.all(
            torch.abs(t12_correct_ious - ious.squeeze(1)) < self.epsilon
        )
        assert all_true

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

@pytest.mark.unit
class TestMeanAveragePrecision:
    EPSILON = 1e-4

    @pytest.mark.boundary
    def test_wrong_pred_boxes_type(self):
        with pytest.raises(TypeError):
            mean_average_precision(
                pred_boxes="not-a-list",
                labels_boxes=[torch.tensor([1, 2, 3, 4, 5, 6])],
                iou_threshold=0.5,
                box_format=BoxFormat.MIDDLE_POINT,
                num_classes=20,
            )

    @pytest.mark.boundary
    def test_wrong_labels_boxes_type(self):
        with pytest.raises(TypeError):
            mean_average_precision(
                pred_boxes=[torch.tensor([1, 2, 3, 4, 5, 6])],
                labels_boxes="not-a-list",
                iou_threshold=0.5,
                box_format=BoxFormat.MIDDLE_POINT,
                num_classes=20,
            )

    @pytest.mark.boundary
    def test_wrong_iou_threshold_type(self):
        with pytest.raises(TypeError):
            mean_average_precision(
                pred_boxes=[torch.tensor([1, 2, 3, 4, 5, 6])],
                labels_boxes=[torch.tensor([1, 2, 3, 4, 5, 6])],
                iou_threshold="not-a-float",
                box_format=BoxFormat.MIDDLE_POINT,
                num_classes=20,
            )

    @pytest.mark.boundary
    def test_wrong_box_format_type(self):
        with pytest.raises(TypeError):
            mean_average_precision(
                pred_boxes=[torch.tensor([1, 2, 3, 4, 5, 6])],
                labels_boxes=[torch.tensor([1, 2, 3, 4, 5, 6])],
                iou_threshold=0.5,
                box_format="not-a-box-format",
                num_classes=20,
            )

    @pytest.mark.boundary
    def test_wrong_num_classes_type(self):
        with pytest.raises(TypeError):
            mean_average_precision(
                pred_boxes=[torch.tensor([1, 2, 3, 4, 5, 6])],
                labels_boxes=[torch.tensor([1, 2, 3, 4, 5, 6])],
                iou_threshold=0.5,
                box_format=BoxFormat.MIDDLE_POINT,
                num_classes="not-an-int",
            )

    @pytest.mark.boundary
    def test_wrong_iou_threshold_value(self):
        with pytest.raises(ValueError):
            mean_average_precision(
                pred_boxes=[torch.tensor([1, 2, 3, 4, 5, 6])],
                labels_boxes=[torch.tensor([1, 2, 3, 4, 5, 6])],
                iou_threshold=1.5,
                box_format=BoxFormat.MIDDLE_POINT,
                num_classes=20,
            )

    @pytest.mark.boundary
    def test_wrong_num_classes_value(self):
        with pytest.raises(ValueError):
            mean_average_precision(
                pred_boxes=[torch.tensor([1, 2, 3, 4, 5, 6])],
                labels_boxes=[torch.tensor([1, 2, 3, 4, 5, 6])],
                iou_threshold=0.5,
                box_format=BoxFormat.MIDDLE_POINT,
                num_classes=-1,
            )

    @pytest.mark.boundary
    def test_wrong_pred_boxes_shape(self):
        with pytest.raises(ValueError):
            mean_average_precision(
                pred_boxes=[torch.tensor([1, 2, 3])],
                labels_boxes=[torch.tensor([1, 2, 3, 4, 5, 6])],
                iou_threshold=0.5,
                box_format=BoxFormat.MIDDLE_POINT,
                num_classes=20,
            )

    @pytest.mark.boundary
    def test_wrong_labels_boxes_shape(self):
        with pytest.raises(ValueError):
            mean_average_precision(
                pred_boxes=[torch.tensor([1, 2, 3, 4, 5, 6])],
                labels_boxes=[torch.tensor([1, 2, 3])],
                iou_threshold=0.5,
                box_format=BoxFormat.MIDDLE_POINT,
                num_classes=20,
            )

    def test_all_correct_one_class(self):
        t1_preds = [
            torch.Tensor([0, 0, 0.9, 0.55, 0.2, 0.3, 0.2]),
            torch.Tensor([0, 0, 0.8, 0.35, 0.6, 0.3, 0.2]),
            torch.Tensor([0, 0, 0.7, 0.8, 0.7, 0.2, 0.2]),
        ]
        t1_targets = [
            torch.Tensor([0, 0, 0.9, 0.55, 0.2, 0.3, 0.2]),
            torch.Tensor([0, 0, 0.8, 0.35, 0.6, 0.3, 0.2]),
            torch.Tensor([0, 0, 0.7, 0.8, 0.7, 0.2, 0.2]),
        ]

        mean_avg_prec = mean_average_precision(
            t1_preds,
            t1_targets,
            iou_threshold=0.5,
            box_format=BoxFormat.MIDDLE_POINT,
            num_classes=1,
        )

        assert mean_avg_prec == pytest.approx(1, abs=self.EPSILON)

    def test_all_correct_batch(self):
        t2_preds = [
            torch.Tensor([1, 0, 0.9, 0.55, 0.2, 0.3, 0.2]),
            torch.Tensor([0, 0, 0.8, 0.35, 0.6, 0.3, 0.2]),
            torch.Tensor([0, 0, 0.7, 0.8, 0.7, 0.2, 0.2]),
        ]
        t2_targets = [
            torch.Tensor([1, 0, 0.9, 0.55, 0.2, 0.3, 0.2]),
            torch.Tensor([0, 0, 0.8, 0.35, 0.6, 0.3, 0.2]),
            torch.Tensor([0, 0, 0.7, 0.8, 0.7, 0.2, 0.2]),
        ]

        mean_avg_prec = mean_average_precision(
            t2_preds,
            t2_targets,
            iou_threshold=0.5,
            box_format=BoxFormat.MIDDLE_POINT,
            num_classes=1,
        )

        assert mean_avg_prec == pytest.approx(1, abs=self.EPSILON)

    def test_all_wrong_class(self):
        t3_preds = [
            torch.Tensor([0, 1, 0.9, 0.55, 0.2, 0.3, 0.2]),
            torch.Tensor([0, 1, 0.8, 0.35, 0.6, 0.3, 0.2]),
            torch.Tensor([0, 1, 0.7, 0.8, 0.7, 0.2, 0.2]),
        ]
        t3_targets = [
            torch.Tensor([0, 0, 0.9, 0.55, 0.2, 0.3, 0.2]),
            torch.Tensor([0, 0, 0.8, 0.35, 0.6, 0.3, 0.2]),
            torch.Tensor([0, 0, 0.7, 0.8, 0.7, 0.2, 0.2]),
        ]

        mean_avg_prec = mean_average_precision(
            t3_preds,
            t3_targets,
            iou_threshold=0.5,
            box_format=BoxFormat.MIDDLE_POINT,
            num_classes=2,
        )

        assert mean_avg_prec == pytest.approx(0.0, abs=self.EPSILON)

    def test_one_inaccurate_box(self):
        t4_preds = [
            torch.Tensor([0, 0, 0.9, 0.15, 0.25, 0.1, 0.1]),
            torch.Tensor([0, 0, 0.8, 0.35, 0.6, 0.3, 0.2]),
            torch.Tensor([0, 0, 0.7, 0.8, 0.7, 0.2, 0.2]),
        ]
    
        t4_targets = [
            torch.Tensor([0, 0, 0.9, 0.55, 0.2, 0.3, 0.2]),
            torch.Tensor([0, 0, 0.8, 0.35, 0.6, 0.3, 0.2]),
            torch.Tensor([0, 0, 0.7, 0.8, 0.7, 0.2, 0.2]),
        ]
        t4_correct_mAP = 5 / 18

        mean_avg_prec = mean_average_precision(
            t4_preds,
            t4_targets,
            iou_threshold=0.5,
            box_format=BoxFormat.MIDDLE_POINT,
            num_classes=1,
        )

        assert mean_avg_prec == pytest.approx(t4_correct_mAP, abs=self.EPSILON)