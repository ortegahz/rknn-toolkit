import os
import shutil
from rknn.api import RKNN

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
                                         dataset='/home/manu/tmp/dataset.txt')
    if ret != 0:
        print('hybrid_quantization_step2 failed!')
        exit(ret)
    print('done')

    # Export RKNN model
    print('--> Export RKNN model')
    ret = rknn.export_rknn('./torch_jit.rknn')
    if ret != 0:
        print('Export RKNN model failed!')
        exit(ret)
    print('done')

    dir_nqa = './normal_quantization_analysis'
    if os.path.exists(dir_nqa):
        shutil.rmtree(dir_nqa)

    # Accuracy analysis
    print('--> Accuracy analysis')
    rknn.accuracy_analysis(inputs='./dataset.txt', output_dir=dir_nqa)
    print('done')

    rknn.release()

