import os

from rknn.api import RKNN

from test_rknn_6_s1 import ONNX_MODEL, DATASET

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

    # Load ONNX model
    print('--> Loading model')
    ret = rknn.load_onnx(model=ONNX_MODEL,
                         outputs=['onnx::Sigmoid_326',
                                  'onnx::Sigmoid_349',
                                  'onnx::Sigmoid_372',
                                  'onnx::Sigmoid_395',
                                  'onnx::Reshape_329',
                                  'onnx::Reshape_352',
                                  'onnx::Reshape_375',
                                  'onnx::Reshape_398'])
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    path_cfg = 'torch_jit.quantization.cfg'
    if os.path.exists(path_cfg):
        os.remove(path_cfg)

    # Hybrid quantization step1
    print('--> hybrid_quantization_step1')
    ret = rknn.hybrid_quantization_step1(dataset=DATASET)
    if ret != 0:
        print('hybrid_quantization_step1 failed!')
        exit(ret)
    print('done')

    # customized_quantize_layers: {Conv_Conv_24_172: float32, Conv_Conv_120_95: float32}

    rknn.release()
