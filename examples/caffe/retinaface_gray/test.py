import numpy as np
import cv2
from rknn.api import RKNN


def show_outputs(outputs):
    output = outputs[0].reshape(-1)
    output_sorted = sorted(output, reverse=True)
    top5_str = 'mobilenet_v2\n-----TOP 5-----\n'
    for i in range(5):
        value = output_sorted[i]
        index = np.where(output == value)
        for j in range(len(index)):
            if (i + j) >= 5:
                break
            if value > 0:
                topi = '{}: {}\n'.format(index[j], value)
            else:
                topi = '-1: 0.0\n'
            top5_str += topi
    print(top5_str)


if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNN()
    
    # Set model config
    print('--> Config model')
    rknn.config(mean_values=[[0.0]], std_values=[[1.0]], target_platform='rv1126')
    print('done')

    # Load caffe model
    print('--> Loading model')
    ret = rknn.load_caffe(model='/media/manu/kingstop/workspace/MXNet2Caffe/model_caffe/retina-symbol.prototxt',
                          proto='caffe',
                          blobs='/media/manu/kingstop/workspace/MXNet2Caffe/model_caffe/retina-symbol.caffemodel')
    if ret != 0:
        print('Load mobilenet_v2 failed! Ret = {}'.format(ret))
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=False, dataset='./dataset.txt')
    if ret != 0:
        print('Build mobilenet_v2 failed!')
        exit(ret)
    print('done')

    # Export RKNN model
    print('--> Export RKNN model')
    ret = rknn.export_rknn('./mobilenet_v2.rknn')
    if ret != 0:
        print('Export mobilenet_v2.rknn failed!')
        exit(ret)
    print('done')

    # Set inputs
    img = cv2.imread('/media/manu/samsung/pics/students.bmp')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')

    # Inference
    print('--> Running model')
    outputs = rknn.inference(inputs=[img])
    # show_outputs(outputs)
    print('done')

    # save outputs
    for save_i in range(len(outputs)):
        save_output = outputs[save_i].flatten()
        np.savetxt('/home/manu/tmp/rknn_output_%s.txt' % save_i, save_output,
                   fmt="%f", delimiter="\n")

    # # Evaluate model performance
    # print('--> Evaluate model performance')
    # perf_results = rknn.eval_perf(inputs=[img])
    # print('done')

    rknn.release()

