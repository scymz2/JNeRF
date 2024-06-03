import jittor as jt
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def pose_spherical(theta, phi, radius):
    trans_t = lambda t : jt.array(np.array([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,t],
        [0,0,0,1]]).astype(np.float32))
    rot_phi = lambda phi : jt.array(np.array([
        [1,0,0,0],
        [0,np.cos(phi),-np.sin(phi),0],
        [0,np.sin(phi), np.cos(phi),0],
        [0,0,0,1]]).astype(np.float32))
    rot_theta = lambda th : jt.array(np.array([
        [np.cos(th),0,-np.sin(th),0],
        [0,1,0,0],
        [np.sin(th),0, np.cos(th),0],
        [0,0,0,1]]).astype(np.float32))
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = jt.array(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    c2w = c2w[:-1, :]
    return c2w

def path_spherical(radius=4.0, phi=0.0, nframe=80):
    poses = [pose_spherical(angle, phi, radius) for angle in np.linspace(-180,180,nframe+1)[:-1]]
    return poses

def draw_coordinate_system(ax, origin, basis, label, color):
    for i, col in enumerate(color):
        ax.quiver(*origin, *basis[:, i], color=col, length=0.5)
        ax.text(*(origin + basis[:, i]), f'{label}{["x", "y", "z"][i]}', color=col)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Draw world coordinate system
world_origin = np.array([0, 0, 0])
world_basis = np.eye(3)
draw_coordinate_system(ax, world_origin, world_basis, 'W', ['r', 'g', 'b'])

# Draw camera coordinate systems along a spherical path
radius = 4.0
phi = -30.0  # -30 degrees
nframe = 8  # Reduce the number of frames for clearer visualization
poses = path_spherical(radius, phi, nframe)

for pose in poses:
    camera_origin = pose[:3, 3].numpy()
    camera_basis = pose[:3, :3].numpy()
    draw_coordinate_system(ax, camera_origin, camera_basis, 'C', ['r', 'g', 'b'])

# Set the aspect ratio of the plot to be equal
ax.set_box_aspect([1,1,1])

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('World and Camera Coordinate Systems (Spherical Path)')

# Set the limits of the plot
ax.set_xlim([-5, 5])
ax.set_ylim([-5, 5])
ax.set_zlim([-5, 5])

plt.show()
