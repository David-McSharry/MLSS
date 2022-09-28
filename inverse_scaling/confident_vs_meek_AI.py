import numpy as np

seventy_oneshot_davinci_smart = np.array([80, 85, 60, 30, 90])
seventy_oneshot_davinci_smart_probs = np.array([45 , 19, 12.5, 6.87, 6.6]) / 100


seventy_oneshot_davinci_dumb = np.array([80, 60, 90, 95, 50])
seventy_oneshot_davinci_dumb_probs = np.array([40 , 25, 9, 6.5, 4.2]) /100




# find the dot product of the above arrays
smart_Evalue = np.dot(seventy_oneshot_davinci_smart, seventy_oneshot_davinci_smart_probs)
dumb_Evalue = np.dot(seventy_oneshot_davinci_dumb, seventy_oneshot_davinci_dumb_probs)


# print the results
print("Smart Evaluate: " + str(smart_Evalue))
print("Dumb Evaluate: " + str(dumb_Evalue))


