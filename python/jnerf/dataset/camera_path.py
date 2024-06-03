import jittor as jt
import numpy as np

def pose_spherical(theta, phi, radius):
    # This matrix translates the camera of t units along the z axis 
    trans_t = lambda t : jt.array(np.array([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,t],
        [0,0,0,1]]).astype(np.float32))
    # This matrix rotates the camera of phi degrees around the x axis
    rot_phi = lambda phi : jt.array(np.array([
        [1,0,0,0],
        [0,np.cos(phi),-np.sin(phi),0],
        [0,np.sin(phi), np.cos(phi),0],
        [0,0,0,1]]).astype(np.float32))
    # This matrix rotates the camera of theta degrees around the y axis
    rot_theta = lambda th : jt.array(np.array([
        [np.cos(th),0,-np.sin(th),0],
        [0,1,0,0],
        [np.sin(th),0, np.cos(th),0],
        [0,0,0,1]]).astype(np.float32))
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = jt.array(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w # This matrix flips the x axis
    c2w = c2w[:-1, :] # This matrix removes the last row
    return c2w

def path_spherical(nframe=160):
    # poses = [pose_spherical(angle, -30.0, 2.0) for angle in np.linspace(-180,180,nframe+1)[:-1]]
    poses = [pose_spherical(angle, -30.0, 2.0) for angle in np.linspace(-180,180,nframe+1)[:-1]]
    return poses
