# %%
from keras.datasets import mnist
from keras.utils import to_categorical
(x_train,y_train),(x_test,y_test)=mnist.load_data()
# 归一化
print("归一化前y_train\n",y_train[0:100],"\n")
# 独热编码 
# y_train=to_categorical(y_train,num_classes=10)
# print("\n归一化前,独热编码y_train:",y_train[0:100],"\n")


x_train=x_train/255.0
x_test=x_test/255.0
#! erro:对标签值归一化后使用to_to_categorical()函数进行独热编码，编码结果是错误的，
#! issue: to_categorical()函数对缩放后的标签值编码结果与缩放前的结果不一致
# y_train=y_train/255.0
# y_test=y_test/255.0

print("\n归一化 y_train:",y_train[0:100],"\n")

y_train=to_categorical(y_train,num_classes=10)
y_test=to_categorical(y_test,num_classes=10)

print("\n归一化后,独热编码y_train:",y_train[0:100],"\n")
# print(x_test[1],"\n")
# print(y_train[0:100])
import matplotlib.pyplot as plt
plt.imshow(x_test[1],cmap="gray")

# %%
from keras.models import Sequential
from keras.layers import Dense,Flatten
from keras.optimizers import RMSprop

# %%
model = Sequential(name="FCN-Net")
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(units=64,activation="relu"))
model.add(Dense(units=32,activation="relu"))
model.add(Dense(units=32,activation="relu"))
model.add(Dense(units=10,activation="softmax"))
model.summary()

# %%
model.compile(loss='categorical_crossentropy',
              metrics=['accuracy'],
              optimizer=RMSprop())

# %%
model_info=model.fit(x_train,
          y_train,
          epochs=10,
          batch_size=16,
          validation_split=0.2)
loss, accuracy = model.evaluate(x_test, y_test)
print (accuracy)

# %%
import matplotlib.pyplot as plt
y=model_info.history["accuracy"]
print(y)
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.plot(range(0,len(model_info.history["accuracy"])),y)
plt.show()

# %%



