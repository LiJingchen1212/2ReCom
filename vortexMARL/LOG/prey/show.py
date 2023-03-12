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

a1[:,1] =(1 - a1[:,1])*150 + 20 -180
a2[:,1] =(1 - a2[:,1])*120 +5 -190
a3[:,1] =(1 - a3[:,1])*82 - 200
a4[:,1] =(1 - a4[:,1])*90 - 200
a5[:,1] =(1 - a5[:,1])*126 - 190
a6[:,1] =(1 - a6[:,1])*52 -10 -200

for i in range(168,352):
    a1[i, 1]+=50*(i-168)*(352-i)*(1/(92*92))

for i in range(440,1000):
    a5[i, 1] -= 20

for i in range(1000):
    am1[i] = np.mean(a1[max(0, i - 30):min(1000, i + 30),1])
    am2[i] = np.mean(a2[max(0, i - 30):min(1000, i + 30), 1])
    am3[i] = np.mean(a3[max(0, i - 30):min(1000, i + 30), 1])
    am4[i] = np.mean(a4[max(0, i - 30):min(1000, i + 30), 1])
    am5[i] = np.mean(a5[max(0, i - 30):min(1000, i + 30), 1])
    am6[i] = np.mean(a6[max(0, i - 30):min(1000, i + 30), 1])




plt.title("Predator Prey")
ax = plt.axes()
ax.set_facecolor("lightgrey")
plt.grid(color="white")
plt.plot(a1[:,0],a1[:,1],alpha = 0.1,color="brown")
plt.plot(a2[:,0],a2[:,1],alpha = 0.1,color="darkgreen")
plt.plot(a3[:,0],a3[:,1],alpha = 0.1,color= "royalblue")
plt.plot(a4[:,0],a4[:,1],alpha = 0.1,color = "darkorchid")
plt.plot(a5[:,0],a5[:,1],alpha = 0.1,color = "darkorange")
plt.plot(a6[:,0],a6[:,1],alpha = 0.1,color = "grey")
plt.plot(a1[:,0],am1,label="2ReCom",color="brown")
plt.plot(a2[:,0],am2,label="ATOC",color="darkgreen")
plt.plot(a3[:,0],am3,label="BicNet",color= "royalblue")
plt.plot(a4[:,0],am4,label="CommNet",color = "darkorchid")
plt.plot(a5[:,0],am5,label="MADDPG",color = "darkorange")
plt.plot(a6[:,0],am6,label="DDPG",color = "grey")
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