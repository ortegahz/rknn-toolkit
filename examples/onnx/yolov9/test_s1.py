import os

import cv2
import numpy as np
from rknn.api import RKNN

ONNX_MODEL = 'modified_modified_yolov9-s-converted-simplify.onnx'
RKNN_MODEL = 'modified_yolov9-s-converted-simplify.rknn'
IMG_PATH = './visi_000000_736_1280.bmp'
DATASET = './dataset.txt'

QUANTIZE_ON = True
SAVE_RESULT = False
PRE_COMPILE = True

BOX_THRESH = 0.5
NMS_THRESH = 0.6
IMG_SIZE = (1280, 736)  # (width, height), such as (1280, 736)

CLASSES = ("fire", "candle_flame", "round_fire",)

if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNN()

    if not os.path.exists(ONNX_MODEL):
        print('model not exist')
        exit(-1)

    # pre-process config
    print('--> Config model')
    rknn.config(reorder_channel='0 1 2',
                mean_values=[[0, 0, 0]],
                std_values=[[255, 255, 255]],
                optimization_level=3,
                target_platform='rv1126',
                output_optimize=1,
                quantize_input_node=QUANTIZE_ON)
    print('done')

    # Load ONNX model
    print('--> Loading model')
    ret = rknn.load_onnx(model=ONNX_MODEL,
                         outputs=['output',
                                  'output1',
                                  'output2',
                                  'output3',
                                  'output4',
                                  'output5',])
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=QUANTIZE_ON, dataset=DATASET, pre_compile=PRE_COMPILE)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export RKNN model
    print('--> Export RKNN model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export model failed!')
        exit(ret)
    print('done')

    if SAVE_RESULT:
        # init runtime environment
        print('--> Init runtime environment')
        ret = rknn.init_runtime()
        # ret = rknn.init_runtime('rk1808', device_id='1808')
        if ret != 0:
            print('Init runtime environment failed')
            exit(ret)
        print('done')

        # Set inputs
        img = cv2.imread(IMG_PATH)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Inference
        print('--> Running model')
        outputs = rknn.inference(inputs=[img])

        # save outputs
        for save_i in range(len(outputs)):
            save_output = outputs[save_i].flatten()
            np.savetxt('/home/manu/tmp/rknn_output_%s.txt' % save_i, save_output, fmt="%f", delimiter="\n")

    rknn.release()
