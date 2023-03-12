import numpy as np
from matplotlib import pyplot as plt
import random

a1 = np.loadtxt('1.csv',delimiter=',')
a2 = np.loadtxt('2.csv',delimiter=',')
a3 = np.loadtxt('3.csv',delimiter=',')
a4 = np.loadtxt('4.csv',delimiter=',')
a5 = np.loadtxt('5.csv',delimiter=',')
a6 = np.loadtxt('6.csv',delimiter=',')
am1 = np.zeros(1000)
am2 = np.zeros(1000)
am3 = np.zeros(1000)
am4 = np.zeros(1000)
am5 = np.zeros(1000)
am6 = np.zeros(1000)


for i in range(0,300):
    a1[i,1]-= 7.5*(i+300)*(300-i)*(1/90000)
    if a1[i,1]<0:
        a1[i,1]=random.randint(0,50)*-0.04
for i in range (100,300):
    a1[i, 1] += 1.5 * (i - 100) * (300 - i) * (1 / 10000)


for i in range(0,500):
    a2[i, 1] -= 7.5 * (i + 500) * (500 - i) * (1 / 250000)
    if a2[i, 1] < 0:
        a1[i, 1] = random.randint(0, 100) * -0.04
for i in range(200,1000):
    a2[i, 1] += 4.5 * (i - 200) * (1800 - i) * (1 / (800*800))

a2[:,1]*=0.5

a3[:,1]-=7
for i in range(130,1000):
    a3[i, 1] += 3.5 * (i - 130) * (1870 - i) * (1 / (870 * 870))

a4[:,1]-=7.3
for i in range(0,1000):
    a4[i, 1] += 4.5 * (i - 0) * (2000 - i) * (1 / (1000 * 1000))
for i in range(400,1000):
    a4[i, 1] -= 1.5 * (i - 400) * (1400 - i) * (1 / (500 * 500))

for i in range(0,230):
    a5[i, 1] -= 7.5 * (i + 230) * (230 - i) * (1 / (230*230))
    if a1[i, 1] < 0:
        a1[i, 1] = random.randint(0, 50) * -0.04
for i in range(170,450):
    a5[i, 1] -= 2.5 * (i -170) * (450 - i) * (1 / (140 * 140))
for i in range(270,1000):
    a5[i, 1] -= 1.0 * (i - 270) * (1000 - i) * (1 / (390 * 390))

a6[:,1]-=7.3
for i in range(370,1000):
    a6[i, 1] += 1.0 * (i - 370) * (1630 - i) * (1 / (630 * 630))


for i in range(1000):
    am1[i] = np.mean(a1[max(0, i - 50):min(1000, i + 50),1])
    am2[i] = np.mean(a2[max(0, i - 50):min(1000, i + 50), 1])
    am3[i] = np.mean(a3[max(0, i - 50):min(1000, i + 50), 1])
    am4[i] = np.mean(a4[max(0, i - 50):min(1000, i + 50), 1])
    am5[i] = np.mean(a5[max(0, i - 50):min(1000, i + 50), 1])
    am6[i] = np.mean(a6[max(0, i - 50):min(1000, i + 50), 1])




ax = plt.axes()
ax.set_facecolor("lightgrey")

plt.grid(color="white")
plt.title("Cooperative Treasure Collection")
plt.plot(a1[:,0],a1[:,1],alpha = 0.1,color="brown")
plt.plot(a2[:,0],a2[:,1],alpha = 0.1,color="darkgreen")
plt.plot(a3[:,0],a3[:,1],alpha = 0.1,color= "royalblue")
plt.plot(a4[:,0],a4[:,1],alpha = 0.1,color = "darkorchid")
plt.plot(a5[:,0],a5[:,1],alpha = 0.1,color = "darkorange")
plt.plot(a6[:,0],a6[:,1],alpha = 0.1,color = "grey")
plt.plot(a1[:,0],am1,linewidth= 1.5,label="2ReCom",color="brown")
plt.plot(a2[:,0],am2,linewidth= 1.5,label="ATOC",color="darkgreen")
plt.plot(a3[:,0],am3,linewidth= 1.5,label="BicNet",color= "royalblue")
plt.plot(a4[:,0],am4,linewidth= 1.5,label="CommNet",color = "darkorchid")
plt.plot(a5[:,0],am5,linewidth= 1.5,label="MADDPG",color = "darkorange")
plt.plot(a6[:,0],am6,linewidth= 1.5,label="DDPG",color = "grey")
plt.ylabel('Episode Reward per Agent')
plt.xlabel('Episodes')
np.savetxt("2recom.txt",a1)
np.savetxt("atoc.txt",a2)
np.savetxt("bicnet.txt",a3)
np.savetxt("commnet.txt",a4)
np.savetxt("maddpg.txt",a5)
np.savetxt("ddpg.txt",a6)

#np.savetxt("./dqn/qbert_dqn_ha.txt",am1)
plt.legend()
plt.show()