import cv2
import pickle
import numpy as np
from rknn.api import RKNN

IMG_PATH = '/home/manu/nfs/tmp/install/rknn_yolov5_demo/model/sylgd_rp.bmp'

if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNN()

    # Load RKNN model
    print('--> Load RKNN model')
    ret = rknn.load_rknn('./torch_jit.rknn')
    if ret != 0:
        print('Load torch_jit.rknn failed!')
        exit(ret)
    print('done')

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
    for save_i in range(len(outputs)):
        save_output = outputs[save_i].flatten()
        np.savetxt('/home/manu/tmp/rknn_output_%s.txt' % save_i, save_output,
                   fmt="%f", delimiter="\n")

    with open('/home/manu/tmp/rknn_sim_outputs.pickle', 'wb') as f:
        pickle.dump(outputs, f)

    rknn.release()
