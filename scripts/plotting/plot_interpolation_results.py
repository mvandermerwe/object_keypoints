import mmint_utils
import numpy as np
import matplotlib.pyplot as plt

cnn_res_fn = "out/results/cnn_res.pkl.gzip"
kp_res_fn = "out/results/kp_res.pkl.gzip"
kp_interp_res_fn = "out/results/kp_int_res.pkl.gzip"

cnn_res = mmint_utils.load_gzip_pickle(cnn_res_fn)
kp_res = mmint_utils.load_gzip_pickle(kp_res_fn)
kp_interp_res = mmint_utils.load_gzip_pickle(kp_interp_res_fn)

z_angles = np.arange(0.0, np.pi / 2.0, (np.pi / 2.0) / 200)
cnn_iou = cnn_res['iou']
kp_iou = kp_res['iou']
kp_interp_iou = kp_interp_res['iou']

plt.plot(z_angles, cnn_iou, label='CNN-Net')
plt.plot(z_angles, kp_iou, label='KP-Net')
plt.plot(z_angles, kp_interp_iou, label='KP-Net (Interp.)')
plt.xlabel("Angle")
plt.ylabel("IoU")
plt.legend()
plt.xlim(0.0, np.pi/2.0)
plt.ylim(0.0, 1.0)

plt.show()
