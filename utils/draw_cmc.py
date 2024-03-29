from matplotlib import pyplot as plt


viper = plt.plot([0,1,2,5,10],[0,22,32,47,60],'ro-',label='VIPeR')
magi = plt.plot([0,1,2,5,8,10,12,15,18,20],[0,83, 90, 96, 98, 98, 99, 100, 100, 100],'go-',label='MAGI')
ilid = plt.plot([0,1,2,5,10],[0,95,96,97,99],'bo-',label='ILID')
plt.legend(framealpha=0.5)
plt.grid(True)
plt.ylabel('Accuracy %')
plt.xlabel('rank')
plt.title('CMC curve')
plt.show()