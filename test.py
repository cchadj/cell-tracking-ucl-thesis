if __name__ == '__main__':
    import sys
    import numpy as np
    import matplotlib.pyplot as plt

    testData = np.array([[0,0], [0.1, 0], [0, 0.3], [-0.4, 0], [0, -0.5]])
    fig, ax = plt.subplots()
    coll = ax.scatter(testData[:,0], testData[:,1], color=["blue"]*len(testData), picker = True, s=[50]*len(testData))
    plt.grid(True)
    plt.axis([-0.5, 0.5, -0.5, 0.5])

    def on_pick(event):
        print(testData[event.ind], "clicked")
        coll._facecolors[event.ind,:] = (1, 0, 0, 1)
        coll._edgecolors[event.ind,:] = (1, 0, 0, 1)
        fig.canvas.draw()

    fig.canvas.mpl_connect('pick_event', on_pick)
    plt.show()
