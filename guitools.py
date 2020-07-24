from matplotlib import pyplot as plt


def scatter_plot_point_selector(points, ax=None, image=None):
    selected_point_indices = []

    def on_point_pick(event):
        LEFT_CLICK = 1
        RIGHT_CLICK = 3
        ind = event.ind
        # event.ind returns a list.
        # even when multiple indices select the first
        ind = ind[0]

        if event.mouseevent.button == RIGHT_CLICK:
            try:
                selected_point_indices.remove(ind)
                col._facecolors[ind, :] = (0, 0, 1, 1)
            except ValueError:
                pass
        elif event.mouseevent.button == LEFT_CLICK:
            selected_point_indices.append(ind)
            col._facecolors[ind, :] = (0, 1, 0, 1)
            # col._edgecolors[ind, :] = (0, 1, 0, 1)

        fig.canvas.draw()

    if ax is None:
        _, ax = plt.subplots()

    fig = plt.gcf()
    if image is not None:
        ax.imshow(image, cmap='gray')
    ax.set_title('Select positions to keep.')

    col = ax.scatter(points[:, 0],
                     points[:, 1],
                     s=100,
                     c=['blue']*len(points),
                     picker=True)
    cid = fig.canvas.mpl_connect('pick_event', on_point_pick)
    fig.canvas.set_window_title(
        'Left click to pick, right remove.\n'
        'Close window to finish.')

    plt.show()
    return selected_point_indices
