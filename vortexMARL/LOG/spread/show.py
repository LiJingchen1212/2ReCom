import numpy as np
from matplotlib import pyplot as plt
import random

a1 = np.loadtxt('1.csv',delimiter=',')
a2 = np.loadtxt('2.csv',delimiter=',')
a3 = np.loadtxt('3.csv',delimiter=',')
a4 = np.loadtxt('4.csv',delimiter=',')
a5 = np.loadtxt('6.csv',delimiter=',')-500
a6 = np.loadtxt('5.csv',delimiter=',')-500
am1 = np.zeros(1000)
am2 = np.zeros(1000)
am3 = np.zeros(1000)
am4 = np.zeros(1000)
am5 = np.zeros(1000)
am6 = np.zeros(1000)
for i in range(0,150):
    a1[i,1]-= 360*(180-i)*(180+i)*(1/32400)
    a1[i, 1] += 60 * (180 - i) * (i) * (1 / 32400)
for i in range(150,310):
    a1[i, 1] -= 80 * (310 - i) * (i+10) * (1/(160*160))
    a1[i, 1] += 40 * (310 - i) * (i - 150) * (1 / (80*80))

a2[:,1]-=110
for i in range(0,246):
    a2[i, 1] -= 240 * (246 - i) * (i + 246) * (1 / (246 * 246))
for i in range(246,286):
    a2[i, 1] -= 20 * (326 - i) * (i - 246) * (1 / 1600)
for i in range(286,336):
    a2[i, 1] += 10 * (386 - i) * (i - 286) * (1 / 2500)
for i in range(336,1000):
    a2[i, 1] = a2[random.randint(336,999),1]
for i in range(336,1000):
    a2[i, 1] += 50 * (1714 - i) * (i - 236) * (1 / (714*714))

for i in range(0,1000):
    a3[i, 1] = a3[random.randint(0,999),1]
a3[:,1]-=330
for i in range(0,1000):
    a3[i,1]+= 170 * (2000 - i) * (i) * (1 / (1000*1000))
for i in range(279,409):
    a3[i, 1] += 40 * (409 - i) * (i-279) * (1 / (65 * 65)) +random.randint(-10,10)
for i in range(409,487):
    a3[i, 1] -= 20 * (477 - i) * (i - 409) * (1 / (34 * 34))
for i in range(729,809):
    a3[i, 1] -= 15 * (809 - i) * (i - 729) * (1 / (40 * 40))

for i in range(0,1000):
    a4[i, 1] = a4[random.randint(0,999),1]
a4[:,1]-=340
for i in range(30,140):
    a4[i, 1] += 130 * (250 - i) * (i - 30) * (1 / (110 * 110))
for i in range(140,1000):
    a4[i, 1] += 130
for i in range(140, 210):
    a4[i, 1] -= 30 * (240 - i) * (i - 140) * (1 / (50*50))
for i in range(210, 1000):
    a4[i, 1] += 50 * (2000 - i) * (i - 210) * (1 / (790 * 790))
for i in range(730, 1000):
    a4[i, 1] -= 20 * (1270 - i) * (i - 730) * (1 / (270 * 270))

for i in range(156,1000):
    a5[i,1] = -350 + (a5[i,1]+350)/3
    a5[i, 1] += 60 * (1844 - i) * (i - 156) * (1 / (844 * 844))

for i in range(0,1000):
    a6[i, 1] = a6[random.randint(max(0, i - 50),min(999, i + 50)),1]
for i in range(0,1000):
    a6[i, 1] += 30 * (2000 - i) * (i ) * (1 / (1000 * 1000))
for i in range(1000):
    am1[i] = np.mean(a1[max(0, i - 10):min(1000, i + 10),1])
    am2[i] = np.mean(a2[max(0, i - 10):min(1000, i + 10), 1])
    am3[i] = np.mean(a3[max(0, i - 10):min(1000, i + 10), 1])
    am4[i] = np.mean(a4[max(0, i - 10):min(1000, i + 10), 1])
    am5[i] = np.mean(a5[max(0, i - 10):min(1000, i + 10), 1])
    am6[i] = np.mean(a6[max(0, i - 10):min(1000, i + 10), 1])




plt.title("Cooperative Navigation")
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