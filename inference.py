from gpiozero import LED
import numpy as np
from time import sleep
import FaBo9Axis_MPU9250
import sys

led0 = LED(17)
led1 = LED(27)
led2 = LED(22)

mpu9250 = FaBo9Axis_MPU9250.MPU9250()
FC_Weight = np.load("Parameters/FC_Weight.npy")
FC_Bias = np.load("Parameters/FC_Bias.npy")
W = np.load("Parameters/W.npy")
U = np.load("Parameters/U.npy")
b = np.load("Parameters/b.npy")
mean = np.load("Parameters/mean.npy")
std = np.load("Parameters/std.npy")
num_hidden = np.load("Parameters/num_hidden.npy")
dpdncy = np.load("Parameters/dpdncy.npy")

h = np.array(np.zeros((num_hidden,dpdncy+1)),dtype=float)

try:
    while True:
        accel = mpu9250.readAccel()
        gyro = mpu9250.readGyro()
        magn = mpu9250.readMagnet()

        #x = np.random.randn(num_hidden)
        x = np.array([accel['x'],accel['y'],accel['z'],gyro['x'],gyro['y'],gyro['z'],magn['x'],magn['y'],magn['z']],dtype=float)
        x = (x - mean)/std
        x = x.reshape((-1,1))

        for i in range(dpdncy,0,-1):
            h[:,i] = np.tanh(np.matmul(np.transpose(W),x) + np.matmul(np.transpose(U),h[:,i-1].reshape(-1,1)) + b.reshape(-1,1)).reshape((num_hidden))
        h[:,0] = np.tanh(np.array(np.matmul(np.transpose(W),x) + b.reshape(-1,1) ,dtype=float)).reshape((num_hidden))
        
        prediction = np.argmax(np.matmul(np.transpose(h[:,-1].reshape(-1,1)),FC_Weight) + FC_Bias)
        if(prediction == 0): led0.on();
        elif(prediction == 1): led1.on();
        elif(prediction == 2): led2.on();

        sleep(float(1)/int(sys.argv[1]))
        led0.off(); led1.off(); led2.off();
        

except KeyboardInterrupt:
    sys.exit()
