import cv2
import numpy as np

NMS_THRESH = 0.3


def nms_boxes(boxes, scores):
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1)
        h1 = np.maximum(0.0, yy2 - yy1)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    return np.array(keep)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)


def parser(feat_cls, feat_reg, h, w, s):
    boxes = []
    scores = []
    classes = []

    for i in range(h):
        for j in range(w):
            _conf_max = sigmoid(np.max(feat_cls[:, i, j]))
            _index = np.argmax(feat_cls[:, i, j])
            if _conf_max < 0.1:
                continue
            _reg = feat_reg[:, i, j]
            _reg_w = np.arange(1, 17)
            l_pred, t_pred, r_pred, b_pred = \
                softmax(_reg[:16]), softmax(_reg[16:32]), softmax(_reg[32:48]), softmax(_reg[48:])
            l, t, r, b = np.dot(l_pred, _reg_w), np.dot(t_pred, _reg_w), np.dot(r_pred, _reg_w), np.dot(b_pred, _reg_w)
            x1 = (j + 0.5 - l) * s
            y1 = (i + 0.5 - t) * s
            x2 = (j + 0.5 + r) * s
            y2 = (i + 0.5 + b) * s
            boxes.append([x1, y1, x2, y2])
            scores.append(_conf_max)
            classes.append(_index)

    return np.array(boxes), np.array(scores), np.array(classes)


num_files = 6
base_path = '/home/manu/tmp/rknn_output_{}.txt'

loaded_outputs = []

for i in range(num_files):
    file_path = base_path.format(i)
    try:
        loaded_output = np.loadtxt(file_path, delimiter="\n")
        loaded_outputs.append(loaded_output)
        print(f"Data loaded from {file_path}: {loaded_output.shape}")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

reg_s8 = loaded_outputs[0].reshape((64, 92, 160))
cls_s8 = loaded_outputs[1].reshape((3, 92, 160))
reg_s16 = loaded_outputs[3].reshape((64, 46, 80))
cls_s16 = loaded_outputs[2].reshape((3, 46, 80))
reg_s32 = loaded_outputs[4].reshape((64, 23, 40))
cls_s32 = loaded_outputs[5].reshape((3, 23, 40))

boxes_all, scores_all, classes_all = [], [], []

# Gather all boxes, scores, and classes
for cls, reg, h, w, s in zip(
        [cls_s8, cls_s16, cls_s32],
        [reg_s8, reg_s16, reg_s32],
        [92, 46, 23],
        [160, 80, 40],
        [8, 16, 32]):
    boxes, scores, classes = parser(cls, reg, h, w, s)
    if len(boxes) > 0:
        boxes_all.append(boxes)
        scores_all.append(scores)
        classes_all.append(classes)

# Merge results
boxes_all = np.vstack(boxes_all)
scores_all = np.hstack(scores_all)
classes_all = np.hstack(classes_all)

# Apply NMS
keep_indices = nms_boxes(boxes_all, scores_all)
boxes_nms = boxes_all[keep_indices]
scores_nms = scores_all[keep_indices]
classes_nms = classes_all[keep_indices]

# Read image
image_path = '/home/manu/tmp/visi_000000_736_1280.bmp'
image = cv2.imread(image_path)

if image is not None:
    # Draw boxes and labels on the image
    for box, score, cls in zip(boxes_nms, scores_nms, classes_nms):
        x1, y1, x2, y2 = map(int, box)
        label = f'Class: {cls} Conf: {score:.2f}'
        color = (0, 255, 0)  # Green color for the box
        thickness = 2
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

    # Display the image with bounding boxes and labels
    cv2.imshow('Image with Bounding Boxes', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Error: Image not loaded. Check the image path.")
