import matplotlib.pyplot as plt
import numpy as np

# # generate two arrays
# x = [0, 1, 2, 3, 4, 5]
# y = [1, -1, 0, 3, 6, 2]

# # save the arrays with numpy as .npy
# with open('test.npy', 'wb') as f:
#     np.save(f, np.array(x))
#     np.save(f, np.array(y))

# load the arrays
with open('plot_data/sincnet-r.npy', 'rb') as f:
    a = np.load(f)
    b = np.load(f)

with open('plot_data/sincnet.npy', 'rb') as f:
    c = np.load(f)
    d = np.load(f)

# plot them
plt.plot(b, a, label='SincNet-R', color='red')
plt.plot(d, c, label='SincNet', color='blue')
plt.xlabel('Epoch')
plt.ylabel('Accuracy(%)')
plt.legend()
plt.show()