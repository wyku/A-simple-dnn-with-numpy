import numpy as np
import matplotlib.pyplot as plt

def tag_entry(entry_x, entry_y):
    if (entry_x ** 2 + entry_y ** 2) ** 0.5 <= 1:
        return 1
    else:
        return 0

def generate_circle_inputs(n_data):
    entry_list = []
    labels = []
    for i in range(n_data):
        x = np.random.uniform(-2, 2)
        y = np.random.uniform(-2, 2)
        label = tag_entry(x, y)
        entry_list.append([x, y])
        labels.append(label)
    return list(zip(entry_list, labels))

def plot_data(entry_list, title):
    colors = []
    data = []
    for entry, label in entry_list:
        if int(label) == 1:
            colors.append('orange')
        else:
            colors.append('blue')
        data.append(entry)
    data = np.array(data)
    
    plt.scatter(data[:, 0], data[:, 1], c=colors)
    plt.title(title)
    plt.show()

if __name__ == '__main__':

    NUMBERS_OF_DATA = 100

    data = generate_circle_inputs(NUMBERS_OF_DATA)
    plot_data(data, 'Demo')