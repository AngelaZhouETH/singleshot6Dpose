import os
import numpy as np
from MeshPly import MeshPly
from utils import *    
import torch


meshname = "../sixd_toolkit/data/zhoum/sofa.ply"

mesh = MeshPly(meshname)
vertices = np.c_[np.array(mesh.vertices), np.ones((len(mesh.vertices), 1))].transpose()
corners3D = get_3D_corners(vertices)
internal_calibration = get_camera_intrinsic()


data_folder = "../sixd_toolkit/output/render/sofa/carpet_1_3/"
gt_filename = data_folder+"gt.yml"

f = open(gt_filename, 'r')

for line in f:
	if line[-2]==':':
		idx = line[0:-2]
		name = ""
		for i in range(0,4-len(idx)):
			name += "0"
		name+=idx
	elif "cam_R_m2c" in line:
		splitted = line[14:-2].split(", ")
		R_gt = np.array([float(i) for i in splitted]).reshape(3, 3)
	elif "cam_t_m2c" in line:
		splitted = line[14:-2].split(", ")
		t_gt = np.array([float(i) for i in splitted]).reshape(3, 1)
		Rt_gt = np.concatenate((R_gt, t_gt), axis=1)
		points3D = np.concatenate((np.array([0,0,0,1]).reshape(4,1), corners3D), axis=1)
		proj_2d_gt = compute_projection(points3D, Rt_gt, internal_calibration)
		proj_2d_gt = np.divide(proj_2d_gt, np.array([640.0, 480.0])[:, np.newaxis])

	elif "obj_bb" in line:
		splitted = line[11:-2].split(", ")
		x_range = float(splitted[2])/640.0
		y_range = float(splitted[3])/480.0
		if not os.path.exists(data_folder+"labels"):
			os.mkdir(data_folder+"labels")
		fw = open(data_folder+"labels/"+name+".txt","w+")
		fw.write("%d %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f" % (0, proj_2d_gt[0,0], proj_2d_gt[1,0], proj_2d_gt[0,1], proj_2d_gt[1,1], proj_2d_gt[0,2], proj_2d_gt[1,2], proj_2d_gt[0,3], proj_2d_gt[1,3], proj_2d_gt[0,4], proj_2d_gt[1,4], proj_2d_gt[0,5], proj_2d_gt[1,5], proj_2d_gt[0,6], proj_2d_gt[1,6], proj_2d_gt[0,7], proj_2d_gt[1,7], proj_2d_gt[0,8], proj_2d_gt[1,8], x_range, y_range))

