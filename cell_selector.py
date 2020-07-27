"""
python estimated_location_keeper -c estimated_coords.csv -v video.avi -o output.csv
"""
from sharedvariables import *
import argparse
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from guitools import ScatterPlotPointSelector
from cell_no_cell import get_frames_from_video
import pathlib
from matplotlib.widgets import Button
from matplotlib import widgets
import tkinter as tk
from tkinter import filedialog


class FrameSelector(object):
    def __init__(self,
                 frames,
                 cell_coordinates,
                 output_file,
                 frame_masks=None,
                 stdev_image=None,
                 vessel_mask=None,
                 ):
        """

        Args:
            frames: NxHxWxC, or NxHxW
            frame_indices:
            cell_coordinates: dict, where points[frame_index] = Nx2. The cell positions for each frame.
        """
        self.fig, self.ax = plt.subplots()
        self.frames = frames
        self.frame_masks = frame_masks
        self.vessel_mask = vessel_mask
        self.cell_coords_dict = cell_coordinates
        self.output_file = output_file

        self.key_idx = 0

        self.frame_idx = None
        self.cur_coords = None
        self.cur_frame = None
        self.cur_mask = None
        self.cur_point_selector = None
        # Holds a point selector object for each frame_idx
        self.point_selector_dict = {}

        self.update()

    def select_frame(self, frame_idx):
        if frame_idx == "":
            frame_idx = self.frame_idx
        self.update(int(frame_idx))

    def next_frame(self, event):
        self.key_idx = (self.key_idx + 1) % len(self.cell_coords_dict)
        self.update()

    def prev_frame(self, event):
        self.key_idx = (self.key_idx - 1) % len(self.cell_coords_dict)
        self.update()

    def update(self, frame_idx=None):

        self.frame_idx = frame_idx
        if frame_idx is None:
            self.frame_idx = list(self.cell_coords_dict)[self.key_idx]
        if self.frame_idx in self.cell_coords_dict:
            self.cur_coords = self.cell_coords_dict[self.frame_idx]

        # Get frame at index and mask at that frame
        # CSV is one indexed while python is 0 indexed, subtract one.
        self.cur_frame = self.frames[self.frame_idx - 1]
        if self.frame_idx not in self.point_selector_dict and self.cur_coords is not None:
            self.point_selector_dict[self.frame_idx] = ScatterPlotPointSelector(self.cur_coords, fig_ax=(self.fig, self.ax))

        if self.cur_point_selector is not None:
            self.cur_point_selector.deactivate()

        if self.frame_idx in self.point_selector_dict:
            self.cur_point_selector = self.point_selector_dict[self.frame_idx]
        else:
            self.cur_point_selector = None

        # apply mask from the frame mask video.
        if self.frame_masks is not None:
            self.cur_mask = self.frame_masks[self.frame_idx - 1]
            self.cur_frame[~self.cur_mask] = 0

        # apply mask from the vessel mask
        if self.vessel_mask is not None:
            self.cur_frame[~self.vessel_mask] = 0

        self.ax.clear()
        self.ax.imshow(self.cur_frame, cmap='gray')

        # self.ax.scatter(self.cur_coords[:, 0], self.cur_coords[:, 1])
        self.ax.set_title(f'Frame {self.frame_idx}')
        if self.cur_point_selector is not None:
            self.cur_point_selector.activate()

        self.fig.canvas.draw_idle()

    def save(self, event):
        output_df = pd.DataFrame(columns=['X', 'Y', 'Slice'])
        for frame_idx in self.cell_coords_dict.keys():
            if frame_idx not in self.point_selector_dict:
                continue
            point_selector = self.point_selector_dict[frame_idx]
            if len(point_selector.selected_point_indices) == 0:
                # when no point selected from that frame
                continue

            blood_cell_positions = self.cell_coords_dict[frame_idx]
            positions_to_keep = blood_cell_positions[point_selector.selected_point_indices, :]

            cur_coords_df = pd.DataFrame({
                'X': positions_to_keep[:, 0],
                'Y': positions_to_keep[:, 1],
                'Slice': np.ones(len(positions_to_keep) * frame_idx)})
            output_df = output_df.append(cur_coords_df)

        output_df.to_csv(self.output_file)
        print(f'Saved output to {self.output_file}')


def dir_path(path):
    try:
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        return path
    except:
        raise argparse.ArgumentTypeError(f'readable_dir:{path} is not a valid path')


def parse_arguments():
    """ Parse the arguments and get the video filename and the coordinates along with the frame indices
    (starting from 1)

    The returned coordinates and frame indices should have the same length.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video', type=argparse.FileType('r'), required=False, nargs=1,
                        help='The video that the estimated locations belong to.')
    parser.add_argument('-m', '--mask-video', type=argparse.FileType('r'), required=False, nargs=1,
                        help='The video with the corresponding masks. Must be of same length as the retinal video.')
    parser.add_argument('-c', '--coords', type=argparse.FileType('r'), required=False, nargs=1,
                        help='The coordinates csv. Must have 3 columns, X, Y, Slice. '
                             'X, Y is the location of the coordinate and Slice is the frame starting from 1')
    parser.add_argument('-o', '--output-directory', type=dir_path, default='.',
                        help='The directory for the output file. The created will have the same name as the video'
                             "with '_selected_coords.csv' appended to it.")

    args = parser.parse_args()

    try:
        csv_filename = args.coords[0].name
    except:
        root = tk.Tk()
        root.withdraw()
        print('Select csv file with blood sell positions.')
        csv_filename = tk.filedialog.askopenfilename(title='Select csv file with blood cell positions',
                                                     filetypes=[('CSV files', ['*.txt', '*.csv'])])

    coordinates_df = pd.read_csv(csv_filename, sep=',')
    coords = coordinates_df[['X', 'Y']].to_numpy()
    frame_indices = coordinates_df['Slice'].to_numpy()

    try:
        video_name = args.video[0].name
    except:
        root = tk.Tk()
        root.withdraw()
        print('Select video.')
        video_name = tk.filedialog.askopenfilename(title='Select video',
                                                   filetypes=[('Video files', ['*.avi',
                                                                               '*.flv',
                                                                               '*.mov',
                                                                               '*.mp4',
                                                                               '*.wmv',
                                                                               '*.qt',
                                                                               '*.mkv'])])
    try:
        masks_video_filename = args.video[0].name
    except:
        root = tk.Tk()
        root.withdraw()
        print('Select masks video.')
        masks_video_filename = tk.filedialog.askopenfilename(title='Select video',
                                                   filetypes=[('Video files', ['*.avi',
                                                                               '*.flv',
                                                                               '*.mov',
                                                                               '*.mp4',
                                                                               '*.wmv',
                                                                               '*.qt',
                                                                               '*.mkv'])])
    return video_name, masks_video_filename, coords, frame_indices, args.output_directory


def main():
    video_filename, masks_video_filename, coords, frame_indices, output_directory = parse_arguments()

    output_csv_filename = os.path.splitext(os.path.basename(video_filename))[0] + '_selected_coords.csv'
    output_csv_filename = os.path.join(output_directory, output_csv_filename)

    frames = get_frames_from_video(video_filename)

    # cell_positions[frame_idx] will contain 2xN_cells_in_frame array for each frame
    cell_positions = {}
    [frame_idxs, idxs] = np.unique(frame_indices, return_index=True)
    for i in range(len(frame_idxs)):
        curr_idx = idxs[i]
        frame_idx = frame_idxs[i]

        if i == len(frame_idxs) - 1:
            cell_positions[frame_idx] = (coords[curr_idx:-1])
        else:
            cell_positions[frame_idx] = coords[curr_idx:idxs[i + 1]]

    frame_masks = None
    if masks_video_filename:
        frame_masks = np.bool8(get_frames_from_video(masks_video_filename)[:, 0])
    callback = FrameSelector(frames,
                             cell_positions,
                             output_csv_filename,
                             frame_masks=frame_masks)

    # fig, ax = plt.subplots(figsize=(2, 1))
    # ax.axes.get_xaxis().set_visible(False)
    # ax.axes.get_yaxis().set_visible(True)
    # ax.axes.get_xaxis().set_ticks([])
    # ax.axes.get_yaxis().set_ticks([])
    # ax.axis('off')
    # for item in [fig, ax]:
    #     item.patch.set_visible(False)

    # rect = [left, bottom, width, height]
    axprev = plt.axes([0.85, 0.65, 0.1, 0.175])
    axtxtbox = plt.axes([0.85, 0.50, 0.1, 0.075])
    axnext = plt.axes([0.85, 0.25, 0.1, 0.175])
    axsave = plt.axes([0.85, 0.1, 0.1, .075])

    bnext = Button(axnext, 'Next')
    bnext.on_clicked(callback.next_frame)

    bsave = Button(axsave, 'Save')
    bsave.on_clicked(callback.save)

    frame_txtbox = widgets.TextBox(axtxtbox, label='Fr idx')
    frame_txtbox.on_submit(callback.select_frame)

    def ensure_number(text):
        try:
            if text is not "":
                n = int(text)
                if n >= len(frames):
                    frame_txtbox.set_val(callback.frame_idx)
        except ValueError:
            frame_txtbox.set_val(callback.frame_idx)

    frame_txtbox.on_text_change(ensure_number)

    bprev = Button(axprev, 'Previous')
    bprev.on_clicked(callback.prev_frame)
    plt.show()


if __name__ == '__main__':
    main()
