import numpy as np
from skimage.io import imshow, show
from LPQIE.src.LPIQE_reconstruction_executor import  LpiqeReconstructionExecutor as recon
from LPQIE.src.LPIQE_qiskit_implementation import  LpiqeQiskit

# img=np.full((16,16), 100, np.uint8)/255
# imshow(img, cmap='gray')
# show()
#
# lpique_impl=LpiqeQiskit()
# executor=recon(lpique_impl, (16, 16), [True, '6a780c166609fa9e4de81bb085725b615135a37caf29e4fff82e9d3e3ba864662469ab94fcaa8165e03318df4aa3887b98799b46b7d5979941524a767a9c0f3b', 'ibm-q-psnc', 'internal', 'default', 'simulator_statevector'])
# for i in range
# executor.execute(img, 100000)
# # imshow(executor.reconstructed_image, cmap='gray')
# # show()
# # dim=executor.difference_image
# # imshow(executor.difference_image, cmap='gray')
# # show()
# print('std= '+str(executor.std_dev()), 'mean= '+str(executor.mean()), 'MSE= '+str(executor.mse()))

lpique_impl=LpiqeQiskit()
executor=recon(lpique_impl, (16, 16), [True, '6a780c166609fa9e4de81bb085725b615135a37caf29e4fff82e9d3e3ba864662469ab94fcaa8165e03318df4aa3887b98799b46b7d5979941524a767a9c0f3b', 'ibm-q-psnc', 'internal', 'default', 'simulator_statevector'])
for px in range(0, 255, 50):
    img = np.full((16, 16), px, np.uint8) / 255
    executor.execute(img, 100000, False)
    print('px='+str(px), 'std= ' + str(executor.std_dev()), 'mean= ' + str(executor.mean()), 'MSE= ' + str(executor.mse()))