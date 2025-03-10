import numpy as np
import torch
import os 
import trimesh
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from tqdm import tqdm


# TODO: maybe add arguments into "__main__" module

# TODO: if the trasformation, especally translation is too large
# there can be need for interpolating frames between original sequences
 
def transform_frames(input_folder, output_folder, translation=[0.8, 0.0, 0.0], theta=30.0):

    """
    Convert HDI4D mesh sequence into transformed sequence.

    args:
    translation: rigid translation for final frame in format (x, y, z);
    theta: the rotation angle around Z axis in degree (not rad)
    """

    translation = np.array(translation).squeeze() # (x, y, z)
    theta = np.radians(theta)  # rotation angle
    rotation = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

    # homogeneous trasformation matrix
    total_transform = np.eye(4)
    total_transform[:3, :3] = rotation
    total_transform[:3, 3] = translation

    # load original data
    original_object = []
    original_left_hand = []
    original_right_hand = []

    object_path = os.path.join(input_folder, 'object_anno/mesh')
    left_hand_path = os.path.join(input_folder, 'hand_anno_vit/mano/left')
    right_hand_path = os.path.join(input_folder, 'hand_anno_vit/mano/right')

    num_frames = len(os.listdir(left_hand_path))

    print("Loading original meshes")

    for index in tqdm(range(num_frames)):

        file_name = f"{index:05d}.obj"

        object_mesh = trimesh.load(os.path.join(object_path, file_name))
        left_hand_mesh = trimesh.load(os.path.join(left_hand_path, file_name))
        right_hand_mesh = trimesh.load(os.path.join(right_hand_path, file_name))

        original_object.append(object_mesh)
        original_left_hand.append(left_hand_mesh)
        original_right_hand.append(right_hand_mesh)



    total_rotation = R.from_matrix(total_transform[:3, :3])
    total_translation = total_transform[:3, 3]

    # compute rotations for each frame by slerp
    slerps = Slerp([0, num_frames], R.concatenate([R.identity(), total_rotation]))
    interp_rots = slerps(range(num_frames))

    # compute transformation for each frame
    frame_transforms = []

    print("Attributing transformations")

    for i in tqdm(range(num_frames)):

        # interpolation ratio
        alpha = i / (num_frames - 1)  
        
        # rotation interpolate
        rot_matrix = interp_rots[i].as_matrix()
        
        
        # translate interpolate
        trans = alpha * total_translation
        
        # formulate transform matrixes
        transform = np.eye(4)
        transform[:3, :3] = rot_matrix
        transform[:3, 3] = trans
        frame_transforms.append(transform)



    os.makedirs(output_folder, exist_ok=True)

    output_object_path = os.path.join(output_folder, 'object_anno/mesh')
    output_left_path = os.path.join(output_folder, 'hand_anno_vit/mano/left')
    output_right_path = os.path.join(output_folder, 'hand_anno_vit/mano/right')

    os.makedirs(output_object_path, exist_ok=True)
    os.makedirs(output_left_path, exist_ok=True)
    os.makedirs(output_right_path, exist_ok=True)

    print("Transforming")

    for i in tqdm(range(num_frames)):
        # get original meshes and transform matrix
        original_object_mesh = original_object[i]
        original_left_mesh = original_left_hand[i]
        original_right_mesh = original_right_hand[i]
        transform = frame_transforms[i]
        
        # compute transformed vertices
        homo_object_verts = np.hstack((original_object_mesh.vertices, np.ones((len(original_object_mesh.vertices), 1))))
        transformed_object_verts = (homo_object_verts @ transform.T)[:, :3]

        homo_left_verts = np.hstack((original_left_mesh.vertices, np.ones((len(original_left_mesh.vertices), 1))))
        transformed_left_verts = (homo_left_verts @ transform.T)[:, :3]

        homo_right_verts = np.hstack((original_right_mesh.vertices, np.ones((len(original_right_mesh.vertices), 1))))
        transformed_right_verts = (homo_right_verts @ transform.T)[:, :3]
        
        # save transformed meshes
        transformed_object_mesh = trimesh.Trimesh(vertices=transformed_object_verts, faces=original_object_mesh.faces)
        transformed_object_mesh.export(os.path.join(output_object_path, f"transformed_{i:05d}.obj"))

        transformed_left_mesh = trimesh.Trimesh(vertices=transformed_left_verts, faces=original_left_mesh.faces)
        transformed_left_mesh.export(os.path.join(output_left_path, f"transformed_{i:05d}.obj"))

        transformed_right_mesh = trimesh.Trimesh(vertices=transformed_right_verts, faces=original_right_mesh.faces)
        transformed_right_mesh.export(os.path.join(output_right_path, f"transformed_{i:05d}.obj"))


if __name__ == "__main__":
    translation = [0.8, 0.0, 0.0]
    theta = 30.0
    input_folder = '1730707919.2094975_bag_pengyang_hand2_hook2'
    output_folder = os.path.join('transformed', input_folder)
    transform_frames(input_folder, output_folder, translation=translation, theta=theta)