import os
import shutil
from rknn.api import RKNN

from test_rknn_6_s1 import DATASET, ACC_ANALYSIS_DATASET, RKNN_MODEL

ACC_ANALYSIS_DIR_OUT = './snapshot_hq'

if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNN()

    # model config
    print('--> Config model')
    rknn.config(reorder_channel='0 1 2',
                mean_values=[[0, 0, 0]],
                std_values=[[255, 255, 255]],
                optimization_level=3,
                target_platform='rv1126',
                output_optimize=1,
                quantize_input_node=True)
    print('done')

    # Hybrid quantization step2
    print('--> hybrid_quantization_step2')
    ret = rknn.hybrid_quantization_step2(model_input='./torch_jit.json',
                                         data_input='./torch_jit.data',
                                         model_quantization_cfg='./torch_jit.quantization.cfg',
                                         dataset=DATASET)
    if ret != 0:
        print('hybrid_quantization_step2 failed!')
        exit(ret)
    print('done')

    # Export RKNN model
    print('--> Export RKNN model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export RKNN model failed!')
        exit(ret)
    print('done')

    if os.path.exists(ACC_ANALYSIS_DIR_OUT):
        shutil.rmtree(ACC_ANALYSIS_DIR_OUT)

    # Accuracy analysis
    print('--> Accuracy analysis')
    rknn.accuracy_analysis(inputs=ACC_ANALYSIS_DATASET, output_dir=ACC_ANALYSIS_DIR_OUT, draw_data_distribute=False)
    print('done')

    rknn.release()

