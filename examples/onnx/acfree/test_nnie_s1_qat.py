import os
import shutil

from rknn.api import RKNN

ONNX_MODEL = '/home/manu/tmp/acfree_quant.onnx'
RKNN_MODEL = '/home/manu/nfs/rv1126/install/rknn_yolov5_demo/model/rv1109_rv1126/acfree.rknn'
IMG_PATH = '/media/manu/samsung/pics/students_lt.bmp'
DATASET = '/home/manu/tmp/dataset.txt'

QUANTIZE_ON = False
ACC_ANALYSIS_ON = False

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
                         outputs=['input.96_quantized'])
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

    dir_out = './snapshot'
    if os.path.exists(dir_out):
        shutil.rmtree(dir_out)

    # Accuracy analysis
    if ACC_ANALYSIS_ON:
        print('--> Accuracy analysis')
        rknn.accuracy_analysis(inputs='./dataset.txt', draw_data_distribute=False)
        print('done')

    rknn.release()
