import os
import pickle

import cv2
import numpy as np
from rknn.api import RKNN

ONNX_MODEL = '/home/manu/tmp/acfree.onnx'
RKNN_MODEL = '/home/manu/nfs/rv1126/install/rknn_yolov5_demo/model/rv1109_rv1126/acfree.rknn'
IMG_PATH = '/media/manu/samsung/pics/students_lt.bmp'
DATASET = './dataset.txt'

QUANTIZE_ON = False
INFER_ON = False

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
                         outputs=['onnx::Sigmoid_237',
                                  'onnx::Sigmoid_260',
                                  'onnx::Sigmoid_283',
                                  'onnx::Reshape_240',
                                  'onnx::Reshape_263',
                                  'onnx::Reshape_286'])
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=QUANTIZE_ON, dataset=DATASET)
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

    if INFER_ON:
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
        # img, ratio, (dw, dh) = letterbox(img, new_shape=(IMG_SIZE[1], IMG_SIZE[0]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Inference
        print('--> Running model')
        outputs = rknn.inference(inputs=[img])

        # save outputs
        if QUANTIZE_ON:
            for save_i in range(len(outputs)):
                save_output = outputs[save_i].flatten()
                np.savetxt('/home/manu/tmp/rknn_output_%s.txt' % save_i, save_output,
                           fmt="%f", delimiter="\n")

            with open('/home/manu/tmp/rknn_sim_outputs.pickle', 'wb') as f:
                pickle.dump(outputs, f)
        else:
            for save_i in range(len(outputs)):
                save_output = outputs[save_i].flatten()
                np.savetxt('/home/manu/tmp/rknn_output_%s_nq.txt' % save_i, save_output,
                           fmt="%f", delimiter="\n")

            with open('/home/manu/tmp/rknn_sim_outputs_nq.pickle', 'wb') as f:
                pickle.dump(outputs, f)

    rknn.release()
