import unittest

import torch
from object_keypoints.metrics import iou_metric


class TestIoUMetric(unittest.TestCase):

    def test_iou_loss_naive(self):
        prediction = torch.ones([1, 64, 64, 64], dtype=torch.float32)
        gt = torch.ones([1, 64, 64, 64], dtype=torch.float32)

        iou_ = iou_metric(prediction, gt)
        self.assertTrue(torch.allclose(iou_, torch.tensor(1.0)))

        prediction = torch.zeros([1, 64, 64, 64])
        iou_ = iou_metric(prediction, gt)
        self.assertTrue(torch.allclose(iou_, torch.tensor(0.0)))

    def test_iou_metric(self):
        gt = torch.zeros([1, 4, 4, 4], dtype=torch.float32)
        gt[0, :2, :2, :2] = 1.0

        prediction = torch.zeros([1, 4, 4, 4], dtype=torch.float32)
        prediction[0, :2, :2, :2] = 1.0
        iou_ = iou_metric(prediction, gt)
        self.assertTrue(torch.allclose(iou_, torch.tensor(1.0)))

        prediction = torch.zeros([1, 4, 4, 4], dtype=torch.float32)
        prediction[0, :2, :2, 0] = 1.0
        iou_ = iou_metric(prediction, gt)
        self.assertTrue(torch.allclose(iou_, torch.tensor(0.5)))

        prediction = torch.zeros([1, 4, 4, 4], dtype=torch.float32)
        prediction[0, :2, 0, 0] = 1.0
        iou_ = iou_metric(prediction, gt)
        self.assertTrue(torch.allclose(iou_, torch.tensor(0.25)))

        prediction = torch.zeros([1, 4, 4, 4], dtype=torch.float32)
        prediction[0, 0, 0, 0] = 1.0
        iou_ = iou_metric(prediction, gt)
        self.assertTrue(torch.allclose(iou_, torch.tensor(0.125)))

        prediction = torch.zeros([1, 4, 4, 4], dtype=torch.float32)
        prediction[0, :3, :3, :3] = 1.0
        iou_ = iou_metric(prediction, gt)
        self.assertTrue(torch.allclose(iou_, torch.tensor(8.0 / 27.0)))

    def test_iou_threshold(self):
        gt = torch.zeros([1, 4, 4, 4], dtype=torch.float32)
        gt[0, :2, :2, :2] = 1.0

        prediction = torch.zeros([1, 4, 4, 4], dtype=torch.float32)
        prediction[0, :2, :2, 0] = 0.4
        prediction[0, :2, :2, 1] = 0.6

        for threshold, iou_gt in zip([0.25, 0.5, 0.75], [1.0, 0.5, 0.0]):
            iou_ = iou_metric(prediction, gt, threshold=threshold)
            self.assertTrue(torch.allclose(iou_, torch.tensor(iou_gt)))

    def test_iou_no_reduce(self):
        gt = torch.zeros([4, 4, 4, 4], dtype=torch.float32)
        gt[:, :2, :2, :2] = 1.0

        prediction = torch.zeros([4, 4, 4, 4], dtype=torch.float32)
        prediction[0, :2, :2, :2] = 1.0
        prediction[1, :2, :2, 0] = 1.0
        prediction[2, :2, 0, 0] = 1.0
        prediction[3, :3, :3, :3] = 1.0
        iou_ = iou_metric(prediction, gt, reduce=False)
        self.assertTrue(torch.allclose(iou_, torch.tensor([1.0, 0.5, 0.25, 8.0 / 27.0])))

        iou_ = iou_metric(prediction, gt, reduce=True)
        self.assertTrue(torch.allclose(iou_, torch.tensor([1.0, 0.5, 0.25, 8.0 / 27.0]).mean()))


if __name__ == '__main__':
    unittest.main()
