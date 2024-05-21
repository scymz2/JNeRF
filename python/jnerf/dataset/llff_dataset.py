
from logging import root
import random
import jittor as jt
from jittor.dataset import Dataset
import os
import json
import cv2
import imageio
from math import pi
from math import tan
from tqdm import tqdm
import numpy as np
from jnerf.utils.registry import DATASETS
from jnerf.dataset.dataset_util import *
from pathlib import Path
from colmapUtils.read_write_model import *
from colmapUtils.read_write_dense import * 


@DATASETS.register_module()
class LLFFDataset():
    def __init__(self, root_dir, batch_size, mode='train', factor=4, llffhold=0, recenter=True, bd_factor=.75, spherify=False, correct_pose=[1,-1,-1], aabb_scale=None, scale=None, offset=None, img_alpha=True,to_jt=True, have_img=True, preload_shuffle=True, use_depth=False):
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.preload_shuffle = preload_shuffle
        scale = None
        offset = None
        self.image_data = []
        self.transforms_gpu=[]
        self.correct_pose=correct_pose
        self.focal_lengths= []
        self.aabb_scale=aabb_scale
        self.have_img=have_img
        self.use_depth=use_depth    # if use depth, the dataset will return depth data
        if self.aabb_scale is None: # Axis Aligned Bounding Box
            print("llff dataset need set aabbscale in config file ,automatically set to 32")
            self.aabb_scale = 32
        self.n_images = 0
        self.img_alpha=img_alpha # if image has alpha channel: opacity
        if scale is None:
            self.scale = NERF_SCALE # model size
        else:
            self.scale = scale
        if offset is None:
            # this offset is usually used to adjust the position of the 3D model. For example, 
            # if the center of your 3D model is not at the origin, you may need to adjust the position of the model by offset to make it at the origin.
            # in this code snippet, the offset is set to [0.5, 0.5, 0.5], which means the model will be shifted by 0.5 units in each direction.
            self.offset = [0.5, 0.5, 0.5]
            self.offset = offset
        self.resolution = [0, 0]
        self.mode = mode
        self.idx_now = 0
        assert isinstance(factor, int)
        # load depth data
        if self.use_depth:
            depth_gts, zero_depth_ids = self.load_colmap_depth(factor=factor, bd_factor=bd_factor)

        poses, bds, i_test, imgdirs = self.load_data(
            factor=factor, recenter=recenter, bd_factor=bd_factor, zero_depth = zero_depth_ids)
        n_images = len(imgdirs)
        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]
        if not isinstance(i_test, list):
            i_test = [i_test]

        if llffhold > 0:
            print('Auto LLFF holdout,', llffhold)
            i_test = np.arange(n_images)[::llffhold]
        i_val = i_test
        i_train = np.array([i for i in np.arange(int(n_images)) if
                            (i not in i_test and i not in i_val)])
        split_dir = os.path.join(root_dir, 'split.json')
        if not os.path.exists(split_dir):
            print("create  {}".format(split_dir))
            splits = {'train': i_train.tolist(), 'test': i_test.tolist(),
                    'val': i_val.tolist()}
            with open(split_dir, 'w')as f:
                json.dump(splits, f)
        assert mode == "train" or mode == "val" or mode == "test"
        if mode == 'train':
            i_select = i_train
        elif mode =='val':
            i_select = i_val
        else:
            i_select = i_test
        
 
        self.construct_dataset(poses, i_select, hwf, imgdirs)
        jt.gc()
        self.image_data = self.image_data.reshape(
            self.n_images, -1, 4).detach()
        # breakpoint()

    def construct_dataset(self, poses, i_select, hwf, imgdirs):
        # poses = poses[i_select]
        self.H = hwf[0]
        self.W = hwf[1]
        f = hwf[2]
        for imgid in tqdm(i_select.tolist()):
            imgdir = imgdirs[imgid]
            img = read_image(imgdir)
            self.image_data.append(img)
            self.n_images += 1
            matrix = np.array(poses[imgid])
            self.transforms_gpu.append(
                self.matrix_nerf2ngp(matrix, self.scale, self.offset))
        self.resolution = [self.W, self.H]
        self.resolution_gpu = jt.array(self.resolution)
        metadata = np.empty([11], np.float32)
        metadata[0] = 0
        metadata[1] = 0
        metadata[2] = 0
        metadata[3] = 0
        metadata[4] = self.W/2/self.W
        metadata[5] = self.H/2/self.H
        focal_length = [f, f]
        self.focal_lengths.append(focal_length)
        metadata[6] = focal_length[0]
        metadata[7] = focal_length[1]

        light_dir = np.array([0, 0, 0])
        metadata[8:] = light_dir
        self.metadata = np.expand_dims(
            metadata, 0).repeat(self.n_images, axis=0)
        assert self.aabb_scale is not None
        aabb_range = (0.5, 0.5)
        self.aabb_range = (
            aabb_range[0]-self.aabb_scale/2, aabb_range[1]+self.aabb_scale/2)
        self.H = int(self.H)
        self.W = int(self.W)
        self.image_data = jt.array(self.image_data)
        self.transforms_gpu = jt.array(self.transforms_gpu)
        self.focal_lengths = jt.array(
            self.focal_lengths).repeat(self.n_images, 1)
        # transpose to adapt Eigen::Matrix memory
        self.transforms_gpu = self.transforms_gpu.transpose(0, 2, 1)
        self.metadata = jt.array(self.metadata)
        if self.img_alpha and self.image_data.shape[-1] == 3:
            self.image_data = jt.concat([self.image_data, jt.ones(
                self.image_data.shape[:-1]+(1,))], -1).stop_grad()
        # generate shuffle index
        self.shuffle_index = jt.randperm(self.H*self.W*self.n_images).detach()
        jt.gc() # garbage collection

    def get_poses(self, images):
        poses = []
        for i in images:
            R = images[i].qvec2rotmat() # get rotation matrix from quaternion (`qvec`)
            t = images[i].tvec.reshape([3,1]) # get translation vector
            bottom = np.array([0,0,0,1.]).reshape([1,4])
            w2c = np.concatenate([np.concatenate([R, t], 1), bottom], 0) # get camera-to-world matrix by concatenating rotation matrix and translation vector
            c2w = np.linalg.inv(w2c) # get world-to-camera matrix by taking the inverse of camera-to-world matrix
            poses.append(c2w)
        return np.array(poses)


    def load_colmap_depth(self, factor, bd_factor):
        data_file = Path(self.basedir) / 'colmap_depth.npy'
    
        images = read_images_binary(Path(self.basedir) / 'sparse' / '0' / 'images.bin')
        points = read_points3d_binary(Path(self.basedir) / 'sparse' / '0' / 'points3D.bin')

        # compute mean reprojection error for all 3D points
        Errs = np.array([point3D.error for point3D in points.values()])
        Err_mean = np.mean(Errs)
        print("Mean Projection Error:", Err_mean)
        
        # get_poses() is used to directly get camera poses from images.bin file, while _load_data() considers image size and scale factor
        poses = self.get_poses(images)
        _, bds_raw, _ = self.load_llff(self.basedir, factor=factor) # factor=8 downsamples original imgs by 8x
        bds_raw = np.moveaxis(bds_raw, -1, 0).astype(np.float32)
        # print(bds_raw.shape)
        # Rescale if bd_factor is provided
        # adjust near and far according to bd_factor, scale_factor adjusts the depth range
        sc = 1. if bd_factor is None else 1./(bds_raw.min() * bd_factor)
        
        near = np.ndarray.min(bds_raw) * .9 * sc
        far = np.ndarray.max(bds_raw) * 1. * sc
        print('near/far:', near, far)

        # initialize data list to store processed depth information
        data_list = []
        zero_depth_ids = []
        for id_im in range(1, len(images)+1):
            depth_list = []
            coord_list = []
            weight_list = []
            for i in range(len(images[id_im].xys)):
                # get 2D coordinates according to image id and get corresponding 3D point id
                point2D = images[id_im].xys[i]
                id_3D = images[id_im].point3D_ids[i]
                if id_3D == -1:
                    continue
                point3D = points[id_3D].xyz
                depth = (poses[id_im-1,:3,2].T @ (point3D - poses[id_im-1,:3,3])) * sc
                # ignore depth values that are not within the near and far range
                if depth < bds_raw[id_im-1,0] * sc or depth > bds_raw[id_im-1,1] * sc:
                    continue
                err = points[id_3D].error
                # calculate weight: the weight calculation method uses the projection error to adjust the contribution of each 3D point
                weight = 2 * np.exp(-(err/Err_mean)**2)
                depth_list.append(depth)
                coord_list.append(point2D/factor)
                weight_list.append(weight)
            if len(depth_list) > 0:
                print(id_im, len(depth_list), np.min(depth_list), np.max(depth_list), np.mean(depth_list))
                data_list.append({"depth":np.array(depth_list), "coord":np.array(coord_list), "error":np.array(weight_list)})
            else:
                zero_depth_ids.append(id_im - 1)
                print(id_im, len(depth_list))
        # json.dump(data_list, open(data_file, "w"))
        np.save(data_file, data_list)
        return data_list, zero_depth_ids


    def load_data(self, factor, recenter, bd_factor, zero_depth=[]):
        poses, bds, imgdirs = self.load_llff(factor)

        # remove images with zero depth
        poses = np.delete(poses, zero_depth, axis=-1)
        bds = np.delete(bds, zero_depth, axis=-1)
        imgdirs = np.delete(imgdirs, zero_depth, axis=-1)
      
        poses = np.concatenate(
            [poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
        poses = np.moveaxis(poses, -1, 0).astype(np.float32)
        # imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
        # images = imgs
        bds = np.moveaxis(bds, -1, 0).astype(np.float32)

        # Rescale if bd_factor is provided
        sc = 1. if bd_factor is None else 1./(bds.min() * bd_factor)
        poses[:, :3, 3] *= sc
        bds *= sc

        if recenter:
            poses = self.recenter_poses(poses)
            pass

        c2w = self.poses_avg(poses)
        # print('Data:')
        # print(poses.shape, bds.shape)
        # find the pose that is closest to the average pose (c2w) and select as the holdout view
        dists = np.sum(np.square(c2w[:3, 3] - poses[:, :3, 3]), -1)
        i_test = np.argmin(dists)
        print('HOLDOUT view is', i_test)

        # images = images.astype(np.float32)
        poses = poses.astype(np.float32)
        return poses, bds, i_test, imgdirs

    def recenter_poses(self, poses):

        poses_ = poses+0
        bottom = np.reshape([0, 0, 0, 1.], [1, 4])
        c2w = self.poses_avg(poses)
        c2w = np.concatenate([c2w[:3, :4], bottom], -2)
        bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
        poses = np.concatenate([poses[:, :3, :4], bottom], -2)

        poses = np.linalg.inv(c2w) @ poses
        poses_[:, :3, :4] = poses[:, :3, :4]
        poses = poses_
        return poses

    def poses_avg(self, poses):

        hwf = poses[0, :3, -1:]

        center = poses[:, :3, 3].mean(0)
        vec2 = normalize(poses[:, :3, 2].sum(0))
        up = poses[:, :3, 1].sum(0)
        c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)

        return c2w

    def load_llff(self, factor=4):
        basedir = self.root_dir
        poses_arr = np.load(os.path.join(self.root_dir, 'poses_bounds.npy'))
        poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
        bds = poses_arr[:, -2:].transpose([1, 0])
        img0 = [os.path.join(self.root_dir, 'images', f) for f in sorted(os.listdir(os.path.join(self.root_dir, 'images')))
                if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
        sh = imageio.imread(img0).shape
        sfx = ''
        if factor is not None:
            sfx = '_{}'.format(factor)
            self._minify(factors=[factor])
        else:
            factor = 1
            assert False, "factor need to provided"

        imgdir = os.path.join(basedir, 'images' + sfx)
        if not os.path.exists(imgdir):
            print(imgdir, 'does not exist, returning')
            return

        imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(
            imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
        
        if poses.shape[-1] != len(imgfiles):
            print('Mismatch between imgs {} and poses {} !!!!'.format(
                len(imgfiles), poses.shape[-1]))
            return

        sh = imageio.imread(imgfiles[0]).shape
        poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
        poses[2, 4, :] = poses[2, 4, :] * 1./factor

        # imgs = [imageio.imread(img) for img in imgfiles]
        # imgs = np.stack(imgs, -1)
        return poses, bds, imgfiles

    def _minify(self, factors=[], resolutions=[]):
        needtoload = True
        basedir = self.root_dir
        for r in factors:
            imgdir = os.path.join(basedir, 'images_{}'.format(r))
            if not os.path.exists(imgdir):
                needtoload = True
        for r in resolutions:
            imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
            if not os.path.exists(imgdir):
                needtoload = True
        if not needtoload:
            print("minify data exist,not needtoload")
            return

        from subprocess import check_output
        imgdir = os.path.join(basedir, 'images')
        imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
        imgs = [f for f in imgs if any(
            [f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
        imgdir_orig = imgdir
        wd = os.getcwd()
        for r in factors + resolutions:
            if isinstance(r, int):
                name = 'images_{}'.format(r)
                resizearg = '{}%'.format(100./r)
            else:
                name = 'images_{}x{}'.format(r[1], r[0])
                resizearg = '{}x{}'.format(r[1], r[0])
            imgdir = os.path.join(basedir, name)
            if os.path.exists(imgdir):
                continue

            print("Minifying llff data to {}".format(imgdir))
            os.makedirs(imgdir)
            check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)
            ext = imgs[0].split('.')[-1]
            args = ' '.join(['mogrify', '-resize', resizearg,
                            '-format', 'png', '*.{}'.format(ext)])
            print(args)
            os.chdir(imgdir)
            check_output(args, shell=True)
            os.chdir(wd)
            if ext != 'png':
                check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
                print('Removed duplicates')
            print('Done')


    def __next__(self):
        """
        get next batch data
        Returns:
            img_ids: image id
            rays_o: rays origin
            rays_d: rays direction
            rgb_target: target rgb
        """
        if self.idx_now+self.batch_size >= self.shuffle_index.shape[0]:
            del self.shuffle_index
            self.shuffle_index = jt.randperm(
                self.n_images*self.H*self.W).detach()
            jt.gc()
            self.idx_now = 0
        img_index = self.shuffle_index[self.idx_now:self.idx_now+self.batch_size]
        img_ids, rays_o, rays_d, rgb_target = self.generate_random_data(
            img_index, self.batch_size)
        self.idx_now += self.batch_size
        return img_ids, rays_o, rays_d, rgb_target

    def generate_random_data(self, index, bs):
        img_id = index//(self.H*self.W)
        img_offset = index % (self.H*self.W)
        focal_length = self.focal_lengths[img_id]
        xforms = self.transforms_gpu[img_id]
        principal_point = self.metadata[:, 4:6][img_id]
        xforms = xforms.permute(0, 2, 1)
        rays_o = xforms[...,  3]
        res = self.resolution_gpu
        x = ((img_offset % self.W)+0.5)/self.W
        y = ((img_offset//self.W)+0.5)/self.H
        xy = jt.stack([x, y], dim=-1)
        rays_d = jt.concat([(xy-principal_point) * res /
                           focal_length, jt.ones([bs, 1])], dim=-1)
        rays_d = jt.normalize(xforms[..., :3].matmul(rays_d.unsqueeze(2)))
        rays_d = rays_d.squeeze(-1)
        rgb_tar = self.image_data.reshape(-1, 4)[index]
        return img_id, rays_o, rays_d, rgb_tar

    def generate_rays_total(self, img_id, H, W):
        H = int(H)
        W = int(W)
        img_size = H*W
        focal_length = self.focal_lengths[img_id]
        xforms = self.transforms_gpu[img_id]
        principal_point = self.metadata[:, 4:6][img_id]
        xy = jt.stack(jt.meshgrid((jt.linspace(0, H-1, H)+0.5)/H, (jt.linspace(0,
                      W-1, W)+0.5)/W), dim=-1).permute(1, 0, 2).reshape(-1, 2)
        # assert H==W
        # xy += (jt.rand_like(xy)-0.5)/H
        xforms = xforms.permute(1, 0)
        rays_o = xforms[:,  3]
        res = jt.array(self.resolution)
        rays_d = jt.concat([(xy-principal_point) * res /
                           focal_length, jt.ones([H*W, 1])], dim=-1)
        rays_d = jt.normalize(xforms[:, :3].matmul(rays_d.unsqueeze(2)))
        rays_d = rays_d.squeeze(-1)
        return rays_o, rays_d

    def generate_rays_total_test(self, img_ids, H, W):
        # select focal,trans,p_point
        focal_length = jt.gather(
            self.focal_lengths, 0, img_ids)
        xforms = jt.gather(self.transforms_gpu, 0, img_ids)
        principal_point = jt.gather(
            self.metadata[:, 4:6], 0, img_ids)
        # rand generate uv 0~1
        xy = jt.stack(jt.meshgrid((jt.linspace(0, H-1, H)+0.5)/H, (jt.linspace(0,
                      W-1, W)+0.5)/W), dim=-1).permute(1, 0, 2).reshape(-1, 2)
        # assert H==W
        # xy += (jt.rand_like(xy)-0.5)/H
        xy_int = jt.stack(jt.meshgrid(jt.linspace(
            0, H-1, H), jt.linspace(0, W-1, W)), dim=-1).permute(1, 0, 2).reshape(-1, 2)
        xforms = xforms.fuse_transpose([0, 2, 1])
        rays_o = jt.gather(xforms, 0, img_ids)[:, :, 3]
        res = jt.array(self.resolution)
        rays_d = jt.concat([(xy-jt.gather(principal_point, 0, img_ids))
                           * res/focal_length, jt.ones([H*W, 1])], dim=-1)
        rays_d = jt.normalize(jt.gather(xforms, 0, img_ids)[
                              :, :, :3].matmul(rays_d.unsqueeze(2)))
        # resolution W H
        # img H W
        rays_pix = ((xy_int[:, 1]) * H+(xy_int[:, 0])).int()
        # rays origin /dir   rays hit point offset
        return rays_o, rays_d, rays_pix

    def generate_rays_with_pose(self, pose, H, W):
        nray = H*W
        pose = self.matrix_nerf2ngp(pose, self.scale, self.offset)
        focal_length = self.focal_lengths[:1].expand(nray, -1)
        xforms = pose.unsqueeze(0).expand(nray, -1, -1)
        principal_point = self.metadata[:1, 4:6].expand(nray, -1)
        xy = jt.stack(jt.meshgrid((jt.linspace(0, H-1, H)+0.5)/H, (jt.linspace(0,
                      W-1, W)+0.5)/W), dim=-1).permute(1, 0, 2).reshape(-1, 2)
        xy_int = jt.stack(jt.meshgrid(jt.linspace(
            0, H-1, H), jt.linspace(0, W-1, W)), dim=-1).permute(1, 0, 2).reshape(-1, 2)
        rays_o = xforms[:, :, 3]
        res = jt.array(self.resolution)
        rays_d = jt.concat([
            (xy-principal_point) * res/focal_length,
            jt.ones([H*W, 1])
        ], dim=-1)
        rays_d = jt.normalize(xforms[:, :, :3].matmul(rays_d.unsqueeze(2)))
        return rays_o, rays_d

    def matrix_nerf2ngp(self, matrix, scale, offset):
        """
        convert matrix from nerf to ngp
        Args:
            matrix: matrix
            scale: scale
            offset: offset
        Returns:
            matrix: matrix
        """
        matrix[:, 0] *= self.correct_pose[0]
        matrix[:, 1] *= self.correct_pose[1]
        matrix[:, 2] *= self.correct_pose[2]
        matrix[:, 3] = matrix[:, 3] * scale + offset
        # cycle
        matrix = matrix[[1, 2, 0]]
        return matrix

    def matrix_ngp2nerf(self, matrix, scale, offset):
        """
        convert matrix from ngp to nerf
        Args:
            matrix: matrix
            scale: scale
            offset: offset
        Returns:
            matrix: matrix
        """
        matrix = matrix[[2, 0, 1]]
        matrix[:, 0] *= self.correct_pose[0]
        matrix[:, 1] *= self.correct_pose[1]
        matrix[:, 2] *= self.correct_pose[2]
        matrix[:, 3] = (matrix[:, 3] - offset) / scale
        return matrix
