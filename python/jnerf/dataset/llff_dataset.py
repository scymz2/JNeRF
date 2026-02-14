
from logging import root
import random
import jittor as jt
from jittor.dataset import Dataset
import os, sys
import json
import cv2
import imageio
from math import pi
from math import tan
from tqdm import tqdm
import numpy as np

# check if it is in debug mode
is_debugging = sys.gettrace() is not None or 'pydevd' in sys.modules

from jnerf.utils.registry import DATASETS
from jnerf.dataset.dataset_util import *


@DATASETS.register_module()
class LLFFDataset():
    def __init__(self, root_dir, batch_size, is_stereo=False, mode='train', factor=4, llffhold=0, recenter=True, bd_factor=.75, spherify=False, correct_pose=[1,-1,-1], aabb_scale=None, scale=None, offset=None, img_alpha=True,to_jt=True, have_img=True, preload_shuffle=True, use_depth=False, depth_rays_prop=0.5, depth_dir='completed_depth', depth_unit_scale=1e-3, **kwargs):
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.preload_shuffle = preload_shuffle
        scale = None
        offset = None
        self.image_data = []
        self.transforms_gpu=[] # transformed poses
        self.correct_pose=correct_pose
        self.focal_lengths= []
        self.aabb_scale=aabb_scale
        self.have_img=have_img
        self.is_stereo = is_stereo
        self.use_depth = use_depth
        self.depth_rays_prop = depth_rays_prop
        self.depth_dir = depth_dir
        self.depth_unit_scale = depth_unit_scale
        self.depth_pose_scale = 1.0
        self.depth_supervision_scale = 1.0
        self.image_paths = []
        self.depth_pool = None
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
        else:
            self.offset = offset
        self.resolution = [0, 0]
        self.mode = mode
        self.idx_now = 0
        assert isinstance(factor, int)

        poses, bds, i_test, imgdirs = self.load_data(
            factor=factor, recenter=recenter, bd_factor=bd_factor)
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
        if self.use_depth:
            self.build_depth_pool()
        jt.gc()
        self.image_data = self.image_data.reshape(
            self.n_images, -1, 4).detach()
        # breakpoint()

    def construct_dataset(self, poses, i_select, hwf, imgdirs):
        # poses = poses[i_select]
        self.H = hwf[0]
        self.W = hwf[1]
        f = hwf[2]
        for imgid in tqdm(np.array(i_select).tolist()):
            imgid = int(imgid)
            if imgid < 0 or imgid >= len(imgdirs):
                continue
            imgdir = imgdirs[imgid]
            img = read_image(imgdir)
            self.image_data.append(img)
            self.image_paths.append(imgdir)
            self.n_images += 1 # number of images
            matrix = np.array(poses[imgid])
            self.transforms_gpu.append(
                self.matrix_nerf2ngp(matrix, self.scale, self.offset))
        self.resolution = [self.W, self.H]
        self.resolution_gpu = jt.array(self.resolution)
        metadata = np.empty([11], np.float32) # metadata for each image [0, 0, 0, 0, 0.5, 0.5, f, f, 0, 0, 0]
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
        # repeat metadata for each image, so it contains the same metadata for each image
        self.metadata = np.expand_dims(
            metadata, 0).repeat(self.n_images, axis=0)
        assert self.aabb_scale is not None
        # initialize aabb_range (bouding box range)
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

    def load_data(self, factor, recenter, bd_factor):
        poses, bds, imgdirs = self.load_llff(factor)

        # remove images with zero depth
        # poses = np.delete(poses, self.zero_depth_ids, axis=-1)
        # bds = np.delete(bds, self.zero_depth_ids, axis=-1)
        # imgdirs = np.delete(imgdirs, self.zero_depth_ids, axis=-1)
      
        poses = np.concatenate(
            [poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
        poses = np.moveaxis(poses, -1, 0).astype(np.float32)
        # imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
        # images = imgs
        bds = np.moveaxis(bds, -1, 0).astype(np.float32)

        # Rescale if bd_factor is provided
        sc = 1. if bd_factor is None else 1./(bds.min() * bd_factor)
        self.depth_pose_scale = sc
        self.depth_supervision_scale = self.depth_pose_scale * self.scale
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

    def build_depth_pool(self):
        depth_root = os.path.join(self.root_dir, self.depth_dir)
        if not os.path.isdir(depth_root):
            raise FileNotFoundError(f"depth directory not found: {depth_root}")

        depth_img_ids = []
        depth_xs = []
        depth_ys = []
        depth_vals = []
        matched_count = 0
        skipped_right = 0

        for local_img_id, img_path in enumerate(self.image_paths):
            # Only left-view images have depth supervision
            if '/left/' not in img_path.replace('\\', '/'):
                skipped_right += 1
                continue

            # Match depth file by stem name (both are .png)
            img_stem = os.path.splitext(os.path.basename(img_path))[0]
            depth_path = os.path.join(depth_root, img_stem + '.png')
            if not os.path.exists(depth_path):
                print(f"[depth] WARNING: depth file not found for {img_stem}")
                continue

            depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth_raw is None:
                print(f"[depth] WARNING: failed to read {depth_path}")
                continue

            depth_raw = depth_raw.astype(np.float32)
            # Resize depth to match training image resolution (self.W x self.H)
            if depth_raw.shape[0] != self.H or depth_raw.shape[1] != self.W:
                depth_resized = cv2.resize(depth_raw, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
            else:
                depth_resized = depth_raw
            # Convert raw depth units to meters, then scale to training coordinate system
            depth_metric = depth_resized * self.depth_unit_scale
            depth_metric = depth_metric * self.depth_supervision_scale

            valid_mask = depth_metric > 0
            ys, xs = np.where(valid_mask)
            if ys.shape[0] == 0:
                continue

            matched_count += 1
            vals = depth_metric[ys, xs]
            depth_img_ids.append(np.full_like(xs, local_img_id, dtype=np.int32))
            depth_xs.append(xs.astype(np.int32))
            depth_ys.append(ys.astype(np.int32))
            depth_vals.append(vals.astype(np.float32))

        if len(depth_vals) == 0:
            raise RuntimeError(f"No valid dense depth pixels found in {depth_root}")

        self.depth_pool = {
            'img_ids': np.concatenate(depth_img_ids, axis=0),
            'xs': np.concatenate(depth_xs, axis=0),
            'ys': np.concatenate(depth_ys, axis=0),
            'depths': np.concatenate(depth_vals, axis=0),
        }
        print(f"[depth] Pool built: {matched_count} images matched, "
              f"{skipped_right} right-view images skipped, "
              f"{self.depth_pool['depths'].shape[0]} valid depth pixels, "
              f"depth range [{self.depth_pool['depths'].min():.4f}, {self.depth_pool['depths'].max():.4f}], "
              f"image size ({self.W}x{self.H})")

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
        np.save(os.path.join(self.root_dir, 'recentered_poses.npy'), poses)
        print('Recentered poses saved!')
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
        poses_arr = np.load(os.path.join(self.root_dir, 'poses_bounds.npy')) # N x 17
        # remove the last two columns of poses_arr and reshape
        poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0]) # 3 x 5 x N
        # Take the last two columns of poses_arr and transpose
        bds = poses_arr[:, -2:].transpose([1, 0]) # 2 x N
        
        # Determine the image directories based on whether stereo mode is enabled
        if self.is_stereo:
            left_imgdir = os.path.join(self.root_dir, 'images', 'left')
            right_imgdir = os.path.join(self.root_dir, 'images', 'right')
            if not os.path.exists(left_imgdir) or not os.path.exists(right_imgdir):
                print(f"Stereo image directories {left_imgdir} or {right_imgdir} do not exist, returning")
                return
            left_img0 = [os.path.join(left_imgdir, f) for f in sorted(os.listdir(left_imgdir))
                        if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
            right_img0 = [os.path.join(right_imgdir, f) for f in sorted(os.listdir(right_imgdir))
                        if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
            # Check if left and right images have the same shape
            if imageio.imread(left_img0).shape != imageio.imread(right_img0).shape:
                print(f"Left image shape {imageio.imread(left_img0).shape} does not match right image shape {imageio.imread(right_img0).shape}, returning")
                return

            img0 = left_img0  # Use left image to determine shape
        else:
            imgdir = os.path.join(self.root_dir, 'images')
            img0 = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))
                    if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
            
        sh = imageio.imread(img0).shape
        sfx = ''
        if factor is not None:
            sfx = '_{}'.format(factor)
            self._minify(factors=[factor])
        else:
            factor = 1
            assert False, "factor need to provided"

        if self.is_stereo:
            left_imgdir = os.path.join(basedir, 'images' + sfx, 'left')
            right_imgdir = os.path.join(basedir, 'images' + sfx, 'right')
            if not os.path.exists(left_imgdir) or not os.path.exists(right_imgdir):
                print(f"Stereo image directories {left_imgdir} or {right_imgdir} do not exist, returning")
                return

            left_imgfiles = [os.path.join(left_imgdir, f) for f in sorted(os.listdir(left_imgdir))
                            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
            right_imgfiles = [os.path.join(right_imgdir, f) for f in sorted(os.listdir(right_imgdir))
                            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
            imgfiles = left_imgfiles + right_imgfiles
        else:
            imgdir = os.path.join(basedir, 'images' + sfx)
            if not os.path.exists(imgdir):
                print(imgdir, 'does not exist, returning')
                return
            imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))
                        if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
            
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

    # def load_llff(self, factor=4, is_stereo=False):
    #     basedir = self.root_dir
    #     poses_arr = np.load(os.path.join(self.root_dir, 'poses_bounds.npy'))
    #     poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
    #     bds = poses_arr[:, -2:].transpose([1, 0])
    #     img0 = [os.path.join(self.root_dir, 'images', f) for f in sorted(os.listdir(os.path.join(self.root_dir, 'images')))
    #             if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
    #     sh = imageio.imread(img0).shape
    #     sfx = ''
    #     if factor is not None:
    #         sfx = '_{}'.format(factor)
    #         self._minify(factors=[factor])
    #     else:
    #         factor = 1
    #         assert False, "factor need to provided"

    #     imgdir = os.path.join(basedir, 'images' + sfx)
    #     if not os.path.exists(imgdir):
    #         print(imgdir, 'does not exist, returning')
    #         return

    #     imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(
    #         imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
        
    #     if poses.shape[-1] != len(imgfiles):
    #         print('Mismatch between imgs {} and poses {} !!!!'.format(
    #             len(imgfiles), poses.shape[-1]))
    #         return

    #     sh = imageio.imread(imgfiles[0]).shape
    #     poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    #     poses[2, 4, :] = poses[2, 4, :] * 1./factor

    #     # imgs = [imageio.imread(img) for img in imgfiles]
    #     # imgs = np.stack(imgs, -1)
    #     return poses, bds, imgfiles

    def _minify(self, factors=[], resolutions=[]):
        needtoload = False
        basedir = self.root_dir

        if self.is_stereo:
            imgdirs = [os.path.join(basedir, 'images', 'left'), os.path.join(basedir, 'images', 'right')]
        else:
            imgdirs = [os.path.join(basedir, 'images')]

        for imgdir in imgdirs:
            for r in factors:
                scaled_imgdir = os.path.join(basedir, 'images_{}'.format(r))
                if not os.path.exists(scaled_imgdir):
                    needtoload = True
            for r in resolutions:
                scaled_imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
                if not os.path.exists(scaled_imgdir):
                    needtoload = True

        if not needtoload:
            print("Minify data exists, not need to load")
            return

        wd = os.getcwd()
        print(f"wd: {wd}")

        for imgdir in imgdirs:
            imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
            imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
            print(f'imgdir: {imgdir}')

            for r in factors + resolutions:
                if isinstance(r, int):
                    name = 'images_{}'.format(r)
                    resizearg = '{}%'.format(100. / r)
                else:
                    name = 'images_{}x{}'.format(r[1], r[0])
                    resizearg = '{}x{}'.format(r[1], r[0])

                scaled_imgdir = os.path.join(basedir, name)
                if not os.path.exists(scaled_imgdir):
                    os.makedirs(scaled_imgdir)

                if self.is_stereo:
                    left_scaled_imgdir = os.path.join(basedir, name, 'left')
                    right_scaled_imgdir = os.path.join(basedir, name, 'right')
                    if not os.path.exists(left_scaled_imgdir):
                        os.makedirs(left_scaled_imgdir)
                    if not os.path.exists(right_scaled_imgdir):
                        os.makedirs(right_scaled_imgdir)

                if imgdir.endswith('left'):
                    target_dir = left_scaled_imgdir
                elif imgdir.endswith('right'):
                    target_dir = right_scaled_imgdir
                else:
                    target_dir = scaled_imgdir
                from subprocess import check_output
                print("Minifying llff data to {}".format(target_dir))
                for img in imgs:
                    check_output('cp {} {}'.format(img, target_dir), shell=True)

                ext = imgs[0].split('.')[-1]
                args = ' '.join(['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)])
                print(args)
                os.chdir(target_dir)
                check_output(args, shell=True)
                os.chdir(wd)
                if ext != 'png':
                    check_output('rm {}/*.{}'.format(target_dir, ext), shell=True)
                    print('Removed duplicates')

                print('Done')



    # def _minify(self, factors=[], resolutions=[]):
    #     needtoload = True
    #     basedir = self.root_dir
    #     for r in factors:
    #         imgdir = os.path.join(basedir, 'images_{}'.format(r))
    #         if not os.path.exists(imgdir):
    #             needtoload = True
    #     for r in resolutions:
    #         imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
    #         if not os.path.exists(imgdir):
    #             needtoload = True
    #     if not needtoload:
    #         print("minify data exist,not needtoload")
    #         return

    #     from subprocess import check_output
    #     imgdir = os.path.join(basedir, 'images')
    #     imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    #     imgs = [f for f in imgs if any(
    #         [f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    #     imgdir_orig = imgdir
    #     wd = os.getcwd()
    #     for r in factors + resolutions:
    #         if isinstance(r, int):
    #             name = 'images_{}'.format(r)
    #             resizearg = '{}%'.format(100./r)
    #         else:
    #             name = 'images_{}x{}'.format(r[1], r[0])
    #             resizearg = '{}x{}'.format(r[1], r[0])
    #         imgdir = os.path.join(basedir, name)
    #         if os.path.exists(imgdir):
    #             continue

    #         print("Minifying llff data to {}".format(imgdir))
    #         os.makedirs(imgdir)
    #         check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)
    #         ext = imgs[0].split('.')[-1]
    #         args = ' '.join(['mogrify', '-resize', resizearg,
    #                         '-format', 'png', '*.{}'.format(ext)])
    #         print(args)
    #         os.chdir(imgdir)
    #         check_output(args, shell=True)
    #         os.chdir(wd)
    #         if ext != 'png':
    #             check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
    #             print('Removed duplicates')
    #         print('Done')


    # def __next__(self):
    #     """
    #     get next batch data
    #     Returns:
    #         img_ids: image id
    #         rays_o: rays origin
    #         rays_d: rays direction
    #         rgb_target: target rgb
    #     """
    #     if self.idx_now+self.batch_size >= self.shuffle_index.shape[0]: # check if the next batch is out of range (if current index + batch size >= total number of images)
    #         del self.shuffle_index
    #         self.shuffle_index = jt.randperm(
    #             self.n_images*self.H*self.W).detach() # generate a new shuffle index from 0 to number of pixels
    #         jt.gc()
    #         self.idx_now = 0
    #     # get image index for current batch from shuffle index
    #     img_index = self.shuffle_index[self.idx_now:self.idx_now+self.batch_size]
    #     # get random data based on image index batch
    #     img_ids, rays_o, rays_d, rgb_target = self.generate_random_data(
    #         img_index, self.batch_size)
    #     self.idx_now += self.batch_size
    #     return img_ids, rays_o, rays_d, rgb_target
    
    def __next__(self):
        """
        get next batch data
        Returns:
            img_ids: image id
            rays_o: rays origin
            rays_d: rays direction
            rgb_target: target rgb
        """

        if not self.use_depth:
            if self.idx_now + self.batch_size >= self.shuffle_index.shape[0]:
                del self.shuffle_index
                self.shuffle_index = jt.randperm(self.n_images * self.H * self.W).detach()
                jt.gc()
                self.idx_now = 0

            img_index = self.shuffle_index[self.idx_now:self.idx_now + self.batch_size]
            img_ids, rays_o, rays_d, rgb_target = self.generate_random_data(img_index, self.batch_size)

            self.idx_now += self.batch_size
            return img_ids, rays_o, rays_d, rgb_target

        n_depth_rays = int(self.batch_size * self.depth_rays_prop)
        n_depth_rays = min(max(n_depth_rays, 1), self.batch_size - 1)
        n_rgb_rays = self.batch_size - n_depth_rays

        if self.idx_now + n_rgb_rays >= self.shuffle_index.shape[0]:
            del self.shuffle_index
            self.shuffle_index = jt.randperm(self.n_images * self.H * self.W).detach()
            jt.gc()
            self.idx_now = 0

        img_index_rgb = self.shuffle_index[self.idx_now:self.idx_now + n_rgb_rays]
        img_ids_rgb, rays_o_rgb, rays_d_rgb, rgb_target = self.generate_random_data(img_index_rgb, n_rgb_rays)

        depth_total = self.depth_pool['depths'].shape[0]
        depth_sample_ids = np.random.randint(0, depth_total, size=(n_depth_rays,), dtype=np.int64)
        img_ids_depth, rays_o_depth, rays_d_depth, depth_target, depth_weights = self.generate_random_data_for_depth(depth_sample_ids, n_depth_rays)

        img_ids = jt.concat([img_ids_rgb, img_ids_depth], dim=0)
        rays_o = jt.concat([rays_o_rgb, rays_o_depth], dim=0)
        rays_d = jt.concat([rays_d_rgb, rays_d_depth], dim=0)

        self.idx_now += n_rgb_rays
        return img_ids, rays_o, rays_d, rgb_target, depth_target, depth_weights




    def generate_random_data(self, index, bs):
        """
        generate random data
        1.generate image id based on index
        2.calculate rays origin and direction
        3.get target rgb

        Args:
            index: index
            bs: batch size
        Returns:
            img_id: image id
            rays_o: rays origin
            rays_d: rays direction
            rgb_tar: target rgb
        """
        img_id = index//(self.H*self.W)         # image index
        img_offset = index % (self.H*self.W)    # pixel offset
        focal_length = self.focal_lengths[img_id]
        xforms = self.transforms_gpu[img_id]
        principal_point = self.metadata[:, 4:6][img_id] # [[0.5, 0.5], .., [0.5, 0.5]]
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
    
        """
        xy: (4096, 2)
        principal_point: (4096, 2)
        res: (2,)
        xf: (4096, 3, 3)
        rays_d: (4096, 3, 1)
        """

    def generate_random_data_for_depth(self, index, bs):
        if isinstance(index, jt.Var):
            ids_np = index.numpy().astype(np.int64)
        else:
            ids_np = np.asarray(index, dtype=np.int64)
        img_ids_np = self.depth_pool['img_ids'][ids_np]
        xs_np = self.depth_pool['xs'][ids_np]
        ys_np = self.depth_pool['ys'][ids_np]
        depths_np = self.depth_pool['depths'][ids_np]

        img_ids_depth = jt.array(img_ids_np).int32()
        depths = jt.array(depths_np).float32()
        weights = jt.ones_like(depths)

        focal_lengths = self.focal_lengths[img_ids_depth]
        xforms = self.transforms_gpu[img_ids_depth]
        principal_points = self.metadata[:, 4:6][img_ids_depth]
        res = self.resolution_gpu

        xs = jt.array(xs_np).float32()
        ys = jt.array(ys_np).float32()
        coords = jt.stack([(xs + 0.5) / self.W, (ys + 0.5) / self.H], dim=-1)

        xforms = xforms.permute(0, 2, 1)
        rays_o = xforms[..., 3]

        rays_d = jt.concat([(coords - principal_points) * res / focal_lengths, jt.ones([bs, 1])], dim=-1)
        rays_d = jt.normalize(xforms[:, :, :3] @ (rays_d.unsqueeze(2))).squeeze(-1)

        return img_ids_depth, rays_o, rays_d, depths, weights

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
