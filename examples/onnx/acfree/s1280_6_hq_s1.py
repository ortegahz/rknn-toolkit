import os
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

    # Load ONNX model
    print('--> Loading model')
    ret = rknn.load_onnx(model='/home/manu/tmp/acfree.onnx',
                         outputs=['/detect/cls_preds.0/Conv_output_0',
                                  '/detect/cls_preds.1/Conv_output_0',
                                  '/detect/cls_preds.2/Conv_output_0',
                                  '/detect/cls_preds.3/Conv_output_0',
                                  '/detect/reg_preds.0/Conv_output_0',
                                  '/detect/reg_preds.1/Conv_output_0',
                                  '/detect/reg_preds.2/Conv_output_0',
                                  '/detect/reg_preds.3/Conv_output_0'])
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    path_cfg = 'torch_jit.quantization.cfg'
    if os.path.exists(path_cfg):
        os.remove(path_cfg)

    # Hybrid quantization step1
    print('--> hybrid_quantization_step1')
    ret = rknn.hybrid_quantization_step1(dataset='/home/manu/tmp/dataset.txt')
    if ret != 0:
        print('hybrid_quantization_step1 failed!')
        exit(ret)
    print('done')

    # Conv_/backbone/ERBlock_4/ERBlock_4.1/block/block.0/rbr_reparam/Conv_203: float32

    # Tips
    print('Please modify *.cfg!')
    print('==================================================================================================')
    print('Modify method:')
    print('Add {layer_name}: {quantized_dtype} to dict of customized_quantize_layers')
    print('If no layer changed, please set {} as empty directory for customized_quantize_layers')
    print('==================================================================================================')
    print('Notes:')
    print('1. The layer_name comes from quantize_parameters, please strip \'@\' and \':xxx\';')
    print('   If layer_name contains special characters, please quote the layer name.')
    print('2. Support quantized_type: asymmetric_affine-u8, dynamic_fixed_point-i8, dynamic_fixed_point-i16, float32.')
    print('3. Please fill in according to the grammatical rules of yaml.')
    print(
        '4. For this model, RKNN Toolkit has provided the corresponding configuration, please directly proceed to step2.')
    print('==================================================================================================')

    rknn.release()

