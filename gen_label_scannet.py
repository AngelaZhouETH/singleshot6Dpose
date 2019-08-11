import os
import json
from plyfile import PlyData, PlyElement
# import pymesh
import numpy as np
from skimage import io
from sklearn.decomposition import PCA
from utils import * 

width = 1296.0
height = 968.0
scans_dir = "/nfs/nas12.ethz.ch/fs1201/infk_ivc_students/cvg-students/zhoum/scannet/scans/"
for scene_id in os.listdir(scans_dir):
	if "scene" not in scene_id:
		continue
	scene_dir = scans_dir+scene_id+"/"
	meshname = scene_dir + scene_id+ "_vh_clean_2.labels.ply"
	if not os.path.exists(meshname):
		continue
	pose_dir = scene_dir + "pose/"
	if not os.path.exists(pose_dir):
		continue
	inst_dir = scene_dir + "instance/"
	if not os.path.exists(inst_dir):
		continue
	segs_file = scene_dir + scene_id + "_vh_clean_2.0.010000.segs.json"
	aggr_file = scene_dir + scene_id + "_vh_clean.aggregation.json"

	# mesh_obj = pymesh.load_mesh(meshname)
	plydata = PlyData.read(meshname)
	scene_mesh = np.zeros(shape=[plydata['vertex'].count, 3], dtype=np.float32)
	scene_mesh[:,0] = plydata['vertex']['x']
	scene_mesh[:,1] = plydata['vertex']['y']
	scene_mesh[:,2] = plydata['vertex']['z']

	obj_list = []
	with open(aggr_file) as json_f:
		data = json.load(json_f)
		for segGroup in data["segGroups"]:
			if segGroup["label"] != "wall" and segGroup["label"] != "floor":
				obj_list.append(segGroup["id"])

	for objectId in obj_list:
		with open(aggr_file) as json_f:
			data = json.load(json_f)
			for segGroup in data["segGroups"]:
				if segGroup["id"] == objectId:
					segments = segGroup["segments"]
					label = segGroup["label"]
					break
		
		vert_idx = []
		with open(segs_file) as json_f:
			data = json.load(json_f)
			for i in range(len(data["segIndices"])):
				if data["segIndices"][i] in segments:
					vert_idx.append(i)

		mesh_vertices = scene_mesh[vert_idx]

		mesh_faces = [[0,1,2]]
		# for face in mesh_obj.faces:
		# 	if face[0] in vert_idx and face[1] in vert_idx and face[2] in vert_idx:
		# 		mesh_faces.append(face)
		# 		break
		
		pca = PCA()
		pca.fit(mesh_vertices)
		idx1 = np.argmax(abs(pca.components_[0:2,2]))
		if pca.components_[idx1,2] >0:
			comp_flag = pca.components_[idx1][0:2]
		else:
			comp_flag = -pca.components_[idx1][0:2]

		# forw_id = 0
		# if np.argmax(abs(pca.components_[:,2])) == 0:
		# 	forw_id = 1
		# forw = pca.components_[forw_id]
		
		pca.fit(mesh_vertices[:,0:2])
		if np.cross(pca.components_[0], pca.components_[1]) < 0:
			pca.components_[0] = -pca.components_[0]
		up = np.array([0.0, 0.0, 1.0])

		if (abs(np.dot(comp_flag,pca.components_[0])) > abs(np.dot(comp_flag,pca.components_[1]))):
			idx2 = 0
		else:
			idx2 = 1
		if np.dot(comp_flag,pca.components_[idx2]) < 0:
			pca.components_ = -pca.components_

		# side = np.cross(up, forw)
		# side /= np.linalg.norm(side)
		# forw = np.cross(side, up)
		# forw /= np.linalg.norm(forw)
		# pca_R = np.concatenate((forw, side, up), axis=0).reshape(3,3)
		pca_R = np.concatenate((pca.components_[0], [0.0], pca.components_[1], [0.0], up), axis=0).reshape(3,3)
		mesh_vertices = mesh_vertices.dot(pca_R.transpose())

		# obj_pymesh = scene_dir+'scannet_pymesh'+str(objectId)+'.ply'
		# newmesh = pymesh.form_mesh(mesh_vertices, np.array(mesh_faces))
		# pymesh.save_mesh(obj_pymesh, newmesh, ascii=True)

		obj_mesh = scene_dir+'scannet_mesh'+str(objectId)+'.ply'
		el = PlyElement.describe(np.array([tuple(row) for row in mesh_vertices], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]), 'vertex')
		PlyData([el], text=True).write(obj_mesh)

		vertices = np.c_[mesh_vertices, np.ones((len(mesh_vertices), 1))].transpose()
		corners3D = get_3D_corners(vertices)
		points3D = np.concatenate((np.mean(corners3D, axis=1).reshape(4,1), corners3D), axis=1)

		internal_calibration = get_camera_intrinsic(2)

		fw_train = open(scene_dir + "train.txt", "w+")
		fw_test = open(scene_dir + "test.txt", "w+")

		# count = 0
		# occlude_count = 0
		for pose_txt in os.listdir(pose_dir):
			f = open(pose_dir+pose_txt)
			Rt = np.linalg.inv(np.array([float(i) for i in f.read().split()]).reshape(4,4))
			# Rt_gt = Rt[0:3, :]
			R = Rt[0:3, 0:3].dot(np.linalg.inv(pca_R))
			t = Rt[0:3, 3].reshape((3, 1))
			Rt_gt = np.concatenate((R, t), axis=1)

			proj_2d_gt = compute_projection(points3D, Rt_gt, internal_calibration)
			if math.isnan(proj_2d_gt[0][0]):
				continue
			if len(proj_2d_gt[2, proj_2d_gt[2,:]<0]) > 4:
				continue
			proj_2d_gt = proj_2d_gt[0:2, :]
			[xrange, yrange] = np.max(proj_2d_gt, axis=1) - np.min(proj_2d_gt, axis=1)
			proj_2d_gt = np.divide(proj_2d_gt, np.array([width, height])[:, np.newaxis])

			if len(proj_2d_gt[:,0][proj_2d_gt[:,0]>0]) != 2:
				continue
			if len(proj_2d_gt[:,0][proj_2d_gt[:,0]<1]) != 2:
				continue
			corner_count = 0
			for i in range(1,9):
				if proj_2d_gt[0,i] > 0 and proj_2d_gt[0,i] < 1 and proj_2d_gt[1,i] > 0 and proj_2d_gt[1,i] < 1:
					corner_count += 1
			if corner_count <= 4:
				continue

			im = io.imread(inst_dir+pose_txt.replace("txt", "png"))
			proj_2d_vertices = compute_projection(vertices, Rt_gt, internal_calibration)[0:2, :].astype(int)
			proj_2d_vertices = proj_2d_vertices[:, proj_2d_vertices[0] < im.shape[1]]
			proj_2d_vertices = proj_2d_vertices[:, proj_2d_vertices[1] < im.shape[0]]


			inst_labels = im[proj_2d_vertices[1,:], proj_2d_vertices[0,:]]
			len(inst_labels[inst_labels==objectId+1])/vertices.shape[1]
			
			if (len(inst_labels[inst_labels==objectId+1])/vertices.shape[1] < 0.6):
				# print (pose_txt)
				# occlude_count+=1
				continue

			fw_train.write(scene_dir +"JPEGImages/"+pose_txt.replace("txt", "jpg")+"\n")
			fw_test.write(scene_dir+"JPEGImages/"+pose_txt.replace("txt", "jpg")+"\n")

			if not os.path.exists(scene_dir+"labels"+str(objectId)+"_" + label):
				os.mkdir(scene_dir+"labels"+str(objectId)+"_" + label)
			fw = open(scene_dir+"labels"+str(objectId)+"_" + label + "/"+pose_txt,"w+")
			fw.write("%d %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f" % (0, proj_2d_gt[0,0], proj_2d_gt[1,0], proj_2d_gt[0,1], proj_2d_gt[1,1], proj_2d_gt[0,2], proj_2d_gt[1,2], proj_2d_gt[0,3], proj_2d_gt[1,3], proj_2d_gt[0,4], proj_2d_gt[1,4], proj_2d_gt[0,5], proj_2d_gt[1,5], proj_2d_gt[0,6], proj_2d_gt[1,6], proj_2d_gt[0,7], proj_2d_gt[1,7], proj_2d_gt[0,8], proj_2d_gt[1,8], xrange/width, yrange/height))
		# 	count += 1
		# print (count)
		# print (occlude_count)
		# print ("****")

