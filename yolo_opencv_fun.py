# -*- coding: utf-8 -*-
###################################################################
# Object detection - YOLOv5_6.0 - OpenCV dnn
# From : https://github.com/samsu2018/yolov5_6.0_opencvdnn_python
# Modify : Sam Su (1, 11, 2022)
##################################################################
import cv2
import argparse
import numpy as np
import time

def get_obj(img, confThreshold, nmsThreshold, net, inpWidth, inpHeight):
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgpath", type=str, default=img, help="image path")
    parser.add_argument('--net', default=net, choices=['yolov5s', 'yolov5l', 'yolov5m', 'yolov5x'])
    parser.add_argument('--confThreshold', default=confThreshold, type=float, help='class confidence')
    parser.add_argument('--nmsThreshold', default=nmsThreshold, type=float, help='nms iou thresh')
    parser.add_argument('--inpWidth', default=inpWidth, type=float, help='nms iou thresh')
    parser.add_argument('--inpHeight', default=inpHeight, type=float, help='nms iou thresh')
    args = parser.parse_args()
    yolonet = yolov5(args.net, 
                     confThreshold=args.confThreshold, 
                     nmsThreshold=args.nmsThreshold, 
                     inpWidth=args.inpWidth,
                     inpHeight=args.inpHeight
                     )

    srcimg = cv2.imread(args.imgpath)
    nms_dets, frame = yolonet.detect(srcimg)

    return nms_dets, frame


class yolov5():
    def __init__(self, yolo_type, confThreshold, nmsThreshold, inpWidth, inpHeight):
        with open('coco.names', 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')
        self.colors = [np.random.randint(0, 255, size=3).tolist() for _ in range(len(self.classes))]
        num_classes = len(self.classes)
        self.anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
        self.nl = len(self.anchors)
        self.na = len(self.anchors[0]) // 2
        self.no = num_classes + 5 
        self.stride = np.array([8., 16., 32.])
        self.inpWidth = inpWidth
        self.inpHeight = inpHeight
        self.net = cv2.dnn.readNetFromONNX(yolo_type + '.onnx')
        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold

    def _make_grid(self, nx=20, ny=20):
        xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
        return np.stack((xv, yv), 2).reshape((1, 1, ny, nx, 2)).astype(np.float32)

    def xywh2xyxy(self,x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
       # y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    def non_max_suppression(self,prediction, conf_thres=0.25,agnostic=False):
        xc = prediction[..., 4] > conf_thres  # candidates
        # Settings
        min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()

        output = [np.zeros((0, 6))] * prediction.shape[0]

        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
            x = x[xc[xi]]  # confidence
            if not x.shape[0]:
                continue
            # Compute conf
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            box = self.xywh2xyxy(x[:, :4])
            # Detections matrix nx6 (xyxy, conf, cls)
            conf = np.max(x[:, 5:], axis=1)
            j = np.argmax(x[:, 5:],axis=1)
            #Convert to array：  x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
            re = np.array(conf.reshape(-1)> conf_thres)
            #Convert to dimension
            conf =conf.reshape(-1,1)
            j = j.reshape(-1,1)
            #numpy concatenate
            x = np.concatenate((box,conf,j),axis=1)[re]
            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            elif n > max_nms:  # excess boxes
                x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence
            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            #Convert to list and using nms of opencv
            boxes = boxes.tolist()
            scores = scores.tolist()
            i = cv2.dnn.NMSBoxes(boxes, scores, self.confThreshold, self.nmsThreshold)
            #i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
            output[xi] = x[i]
        return output

    def letterbox(self,im, new_shape=(640, 640), color=(114, 114, 114), auto=True,scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        # if not scaleup:  # only scale down, do not scale up (for better val mAP)
        #     r = min(r, 1.0)
        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        # if auto:  # minimum rectangle
        #     dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        dw /= 2  # divide padding into 2 sides
        dh /= 2
        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
       
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, ratio, (dw, dh)

    def drawPred(self, frame, classId, conf, left, top, right, bottom):
        # Draw a bounding box.
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), thickness=4)

        label = '%.2f' % conf
        label = '%s:%s' % (self.classes[classId], label)

        # Display the label at the top of the bounding box
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        # cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine), (255,255,255), cv.FILLED)
        cv2.putText(frame, label, (left, top + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)
        return frame

    def detect(self, srcimg): 
        frame = srcimg.copy() # Sam think this is frame.
        im, ratio, wh = self.letterbox(srcimg, self.inpWidth, stride=self.stride, auto=False)
        # Sets the input to the network
        blob = cv2.dnn.blobFromImage(im, 1 / 255.0,(self.inpWidth, self.inpHeight),[0,0,0],swapRB=True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.net.getUnconnectedOutLayersNames())[0]
        # NMS
        pred = self.non_max_suppression(outs, self.confThreshold,agnostic=False)
        # get bbox and draw box
        nms_dets = []
        for i in pred[0]:
            left = int((i[0] - wh[0])/ratio[0])
            top = int((i[1]-wh[1])/ratio[1])
            width = int((i[2] - wh[0])/ratio[0])
            height = int((i[3]-wh[1])/ratio[1])
            conf = i[4]
            classId = int(i[5])
            frame = self.drawPred(frame, classId, conf, left, top, width, height)
            label = str(self.classes[classId])
            nms_dets.append([label, conf, left, top, width, height, classId])
        return nms_dets, frame


# =============================================================================
# The following main functions are used for standalong testing
# =============================================================================
if __name__ == "__main__":
    imgpath = 'bus.jpg'
    net = '../../yolov5_6.0_opencvdnn_python/yolov5s'
    confThreshold = '0.5'
    nmsThreshold = '0.5'
    inpWidth = 640
    inpHeight = 640
    tStart = time.time()
    dets, frame = get_obj(imgpath, confThreshold, nmsThreshold, net, inpWidth, inpHeight)
    print(dets)
    cv2.imwrite('output.jpg', frame)
    print('Story the result to output.jpg')
    print('Spend time:{}'.format(time.time()-tStart))
