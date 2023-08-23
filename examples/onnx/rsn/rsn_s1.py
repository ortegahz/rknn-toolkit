import os
import shutil

from rknn.api import RKNN

ONNX_MODEL = '/home/manu/tmp/iter-96000.onnx'
RKNN_MODEL = '/home/manu/nfs/rv1126/install/rknn_yolov5_demo/model/rv1109_rv1126/iter-96000.rknn'
DATASET = './dataset.txt'
ACC_ANALYSIS_DIR_OUT = './snapshot'
ACC_ANALYSIS_DATASET = './dataset_rsn.txt'

QUANTIZE_ON = False
ACC_ANALYSIS_ON = False
PRE_COMPILE_ON = False

if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNN()

    if not os.path.exists(ONNX_MODEL):
        print('model not exist')
        exit(-1)

    # pre-process config
    print('--> Config model')
    rknn.config(reorder_channel='0 1 2',
                mean_values=[[103.5300, 116.2800, 123.6750]],
                std_values=[[57.3750, 57.1200, 58.3950]],
                optimization_level=3,
                target_platform='rv1126',
                output_optimize=1,
                quantize_input_node=QUANTIZE_ON)
    print('done')

    # Load ONNX model
    print('--> Loading model')
    ret = rknn.load_onnx(model=ONNX_MODEL,
                         outputs=['res'])
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=QUANTIZE_ON, dataset=DATASET, pre_compile=PRE_COMPILE_ON)
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

    # Accuracy analysis
    if ACC_ANALYSIS_ON:
        dir_out = ACC_ANALYSIS_DIR_OUT
        if os.path.exists(dir_out):
            shutil.rmtree(dir_out)
        print('--> Accuracy analysis')
        rknn.accuracy_analysis(inputs=ACC_ANALYSIS_DATASET, draw_data_distribute=False)
        print('done')

    rknn.release()
