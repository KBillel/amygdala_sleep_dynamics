from bk import io

import matplotlib.pyplot as plt


if __name__ == '__main__':
    plt.ion()
    transitions = io.load_shelve('processed_data/transitions')