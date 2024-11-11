# %%
from keras.datasets import mnist
from keras.utils import to_categorical
(x_train,y_train),(x_test,y_test)=mnist.load_data()
print(x_test[1])

# %%
import matplotlib.pyplot as plt
plt.imshow(x_test[1],cmap="gray")

# %%
