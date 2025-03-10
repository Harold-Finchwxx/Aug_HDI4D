import trimesh
import numpy as np
import pyvista as pv
import os
import torch
import glob

# TODO: encapsulate code below into a function
# with __name__ module
# initialize arguments
input_folder = 'transformed/1730707919.2094975_bag_pengyang_hand2_hook2'
object_path = os.path.join(input_folder, 'object_anno/mesh')
left_path = os.path.join(input_folder, 'hand_anno_vit/mano/left')
right_path = os.path.join(input_folder, 'hand_anno_vit/mano/right')

output_folder = input_folder
output_path = os.path.join(output_folder, 'visualization.mp4')

fps = 20
color_map = {
    'object' : 'red',
    'left' : 'blue',
    'right' : 'green'
}

# set up visual window
plotter = pv.Plotter(window_size=(1920, 1088))
plotter.open_movie(output_path, framerate=fps, quality=10)
plotter.camera.azimuth = 0

# load meshes
object_frames = sorted(glob.glob(os.path.join(object_path, "transformed_*.obj")))
left_frames = sorted(glob.glob(os.path.join(left_path, "transformed_*.obj")))
right_frames = sorted(glob.glob(os.path.join(right_path, "transformed_*.obj")))

for i in range(len(object_frames)):

    #plotter.clear()

    object_mesh = pv.read(object_frames[i])
    left_mesh = pv.read(left_frames[i])
    right_mesh = pv.read(right_frames[i])

    plotter.add_mesh(
        object_mesh,
        color=color_map['object'],
        show_edges=True,
        opacity=1.0,
        smooth_shading=True,
        name='object'
    )

    plotter.add_mesh(
        left_mesh,
        color=color_map['left'],
        show_edges=True,
        opacity=1.0,
        smooth_shading=True,
        name='left hand'
    )

    plotter.add_mesh(
        right_mesh,
        color=color_map['right'],
        show_edges=True,
        opacity=1.0,
        smooth_shading=True,
        name='right hand'
    )

    plotter.add_text(
        f"Frame rate: {fps} fps",
        position="upper_left",
        font_size=10
    )

    # fixed view window (optional)
    # TODO: test whether the fixed view work well or not 
    if i == 0:
        plotter.view_isometric()
        plotter.camera.up = (0.0, 1.0, 0.0)
        '''plotter.camera_position = [
        (0.5, 0.5, 0.5),  # position
        (0.0, 0.0, 0.0),  # focus
        (0.0, 0.0, 1.0)   # up orientation
        ]'''
        print(f"initial camera position:\n{plotter.camera_position}")
        plotter.camera_set = True

    # add assistant coordinate label
    plotter.add_axes(
    line_width=5,
    xlabel="X", 
    ylabel="Y", 
    zlabel="Z",
    labels_off=False
    )

    # add orientation label
    plotter.add_text(
        "Z-up coordinate",
        position='upper_right',
        font_size=10
    )

    plotter.write_frame()

plotter.close()
    