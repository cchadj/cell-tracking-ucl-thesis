from matplotlib import pyplot as plt
import numpy as np


def rgba(r, g, b, a=1):
    return r / 255, g / 255, b / 255, a


class ScatterPlotPointSelector(object):
    def __init__(self, points, fig_ax=None, image=None,
                 selected_color=rgba(0, 255, 0),
                 unselected_color=rgba(20, 20, 230)):
        self.fig_ax = fig_ax

        if self.fig_ax is None:
            self.fig_ax = plt.subplots()

        self.fig, self.ax = self.fig_ax
        self.image = image
        self.selected_point_indices = []

        self.cid = None
        self.col = None

        self.is_activated = False
        self.points = points
        self.selected_color = selected_color
        self.unselected_color = unselected_color
        # self.fig.canvas.set_window_title(
        #     'Left click to pick, right remove.\n'
        #     'Close window to finish.')

    def activate(self):
        if self.is_activated:
            raise Exception('Make sure that PointSelector is deactivated before activating')
        self.cid = self.fig.canvas.mpl_connect('pick_event', self.on_point_pick)
        self.col = self.ax.scatter(self.points[:, 0],
                                   self.points[:, 1],
                                   s=100,
                                   c=[self.unselected_color] * len(self.points),
                                   picker=True)
        self._reinit_selected_points()
        self.is_activated = True

    def deactivate(self):
        if not self.is_activated:
            raise Exception('Make sure that PointSelector is activated before deactivating')
        self.fig.canvas.mpl_disconnect(self.cid)
        self.cid = None
        # remove the scatter plot from the axis
        self.col.remove()
        self.col = None
        self.is_activated = False

    def _reinit_selected_points(self):
        """ Give selected_color to selected points when going from deactivated state to activated state """
        try:
            for ind in self.selected_point_indices:
                # noinspection PyProtectedMember
                self.col._facecolors[ind, :] = self.selected_color
                print(ind)
        except ValueError:
            print('Make sure that self.col scatter plot is created before changing the face colors.')
        self.fig.canvas.draw_idle()

    def on_point_pick(self, event):
        LEFT_CLICK = 1
        RIGHT_CLICK = 3
        ind = event.ind
        # event.ind returns a list.
        # even when multiple indices select the first
        ind = ind[0]

        if event.mouseevent.button == RIGHT_CLICK:
            try:
                self.selected_point_indices.remove(ind)
                # noinspection PyProtectedMember
                self.col._facecolors[ind, :] = self.unselected_color
            except ValueError:
                pass
        elif event.mouseevent.button == LEFT_CLICK:
            self.selected_point_indices.append(ind)
            # noinspection PyProtectedMember
            self.col._facecolors[ind, :] = self.selected_color

        self.fig.canvas.draw_idle()


def scatter_plot_point_selector(points, ax=None, image=None):
    selected_point_indices = []

    if ax is None:
        _, ax = plt.subplots()

    fig = plt.gcf()
    if image is not None:
        ax.imshow(image, cmap='gray')
    ax.set_title('Select positions to keep.')

    point_selector = ScatterPlotPointSelector(points, fig_ax=(fig, ax))
    point_selector.activate()
    plt.show()
    return point_selector.selected_point_indices


if __name__ == '__main__':
    points = np.random.randint(0, 10, size=(20, 2))
    points_selected_indices = scatter_plot_point_selector(points)

    print(points_selected_indices)
