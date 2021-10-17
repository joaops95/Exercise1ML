import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
  
# create data
x = [1,2,3,4,5]
y = [3,3,3,3,3]
  


with open("results.json") as jsonFile:
    jsonObject = json.load(jsonFile)
    jsonFile.close()


    

fig, ax1 = plt.subplots()


for key in jsonObject.keys():
# print(jsonObject)
    data = jsonObject[key]['results']
    
    x = np.arange(1, len(data)+1, 1)
    y_data = []
    x_label = []
    for item in data:
        y_data.append(item['reward'])
        x_label.append(item['epoch'])

    print(y_data)
    ax1.plot(x, y_data, label = f"Episode {key}")

    # ax1.set_xticks(x_label)

plt.setp(ax1.get_xticklabels(), rotation=70)

ax1.set_xlabel('Epochs', fontsize=18)
ax1.set_ylabel('Total rewards', fontsize=18)

ax1.legend()
plt.show()