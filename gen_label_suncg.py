import os
import json
import pymesh
import numpy as np
from MeshPly import MeshPly
from utils import *
import sys 
sys.path.append('../sixd_toolkit/pysixd')
import transform
import shutil


width = 640.0
height = 480.0
modelId = 67
meshname = "../Data_raw/object/" + str(modelId)+"/"+str(modelId)+".ply"
scene_dir = "../Data_raw/house/f955987951cbe45d3422d993bbf6bd3c/"
test_id = 0

#convert obj to ply
if (not os.path.isfile(meshname)):
# if True:
	mesh_obj = pymesh.load_mesh(meshname.replace(".ply",".obj"))
	mesh_obj.add_attribute("vertex_normal")
	nvert = int(mesh_obj.vertices.size/3)
	texture = np.zeros((nvert,2))
	vertexNormal = np.zeros((nvert,3))

	objFile = open(meshname.replace(".ply",".obj"), 'r')
	textureList = []
	normalList = []
	for line in objFile:
		split = line.split()
		#if blank line, skip
		if not len(split):
			continue
		if split[0] == "vn":
			normalList.append(list(map(float, split[1:])))
		elif split[0] == "vt":
			textureList.append(list(map(float, split[1:])))

		elif split[0] == "f":
			for face in split[1:]:
				splitted = face.split("/")
				texture[int(splitted[0])-1] = textureList[int(splitted[1])-1]
				vertexNormal[int(splitted[0])-1] = normalList[int(splitted[2])-1]

	texture = texture - np.floor(texture)

	# newmesh = pymesh.form_mesh(mesh_obj.vertices*100, mesh_obj.faces)
	mesh_obj.add_attribute("nx")
	mesh_obj.add_attribute("ny")
	mesh_obj.add_attribute("nz")
	mesh_obj.add_attribute("texture_u")
	mesh_obj.add_attribute("texture_v")
	# mesh_obj.set_attribute("nx", mesh_obj.get_vertex_attribute("vertex_normal")[:,0]);
	# mesh_obj.set_attribute("ny", mesh_obj.get_vertex_attribute("vertex_normal")[:,1]);
	# mesh_obj.set_attribute("nz", mesh_obj.get_vertex_attribute("vertex_normal")[:,2]);
	mesh_obj.set_attribute("nx", np.array(vertexNormal)[:,0]);
	mesh_obj.set_attribute("ny", np.array(vertexNormal)[:,1]);
	mesh_obj.set_attribute("nz", np.array(vertexNormal)[:,2]);
	mesh_obj.set_attribute("texture_u", np.array(texture)[:,0])
	mesh_obj.set_attribute("texture_v", np.array(texture)[:,1])


	# pymesh.save_mesh_raw(meshname, newmesh.vertices, newmesh.faces, newmesh.voxels)
	pymesh.save_mesh(meshname, mesh_obj, "nx", "ny", "nz", "texture_u", "texture_v", ascii=True)

objRT_list = []
num_objs = 0
with open(scene_dir+"house.json") as json_file:
	data = json.load(json_file)
	for level in data['levels']:
		for node in level['nodes']:
			# for keys in node:
			if 'modelId' in node and node['modelId'] == str(modelId):
				objRT =  np.array(node['transform'])
				num_objs+=1
				objRT_list.append(objRT)
print (num_objs)
objRT = objRT_list[test_id]
objRT.resize(4, 4)

mesh = MeshPly(meshname)
vertices = np.c_[np.array(mesh.vertices), np.ones((len(mesh.vertices), 1))].transpose()
corners3D = get_3D_corners(vertices)
points3D = np.concatenate((np.array([0,0,0,1]).reshape(4,1), corners3D), axis=1)
internal_calibration = get_camera_intrinsic()

f = open(scene_dir+"outputcamerafile", 'r')
idx = 0
for line in f:
	camera = np.array([float(i) for i in line.split()])
	forw = camera[3:6] # Forward direction
	forw /= np.linalg.norm(forw)
	up = camera[6:9] # Up direction
	side = np.cross(forw, up) # Side direction
	if np.count_nonzero(side) == 0:
	    # f and u are parallel, i.e. we are looking along or against Z axis
	    side = np.array([1.0, 0.0, 0.0])
	side /= np.linalg.norm(side)
	up = np.cross(side, forw) # Recompute up
	R = np.array([[side[0], side[1], side[2]],
	              [up[0], up[1], up[2]],
	              [-forw[0], -forw[1], -forw[2]]])

	# Convert from OpenGL to OpenCV coordinate system
	R_yz_flip = transform.rotation_matrix(math.pi, [1, 0, 0])[:3, :3]
	R = R_yz_flip.dot(R)

	# # Translation vector
	t = -R.dot(camera[0:3].reshape((3, 1)))
	Rt_gt = np.concatenate((R, t), axis=1)
	proj_2d_gt = compute_projection(points3D, Rt_gt.dot(objRT.transpose()), internal_calibration)
	[xrange, yrange] = np.max(proj_2d_gt, axis=1) - np.min(proj_2d_gt, axis=1)	
	proj_2d_gt = np.divide(proj_2d_gt, np.array([width, height])[:, np.newaxis])

	name = ""
	for i in range(0,6-len(str(idx))):
		name += "0"
	name+=str(idx)
	if not os.path.exists(scene_dir+"labels"):
		os.mkdir(scene_dir+"labels")
	fw = open(scene_dir+"labels/"+name+"_color.txt","w+")
	fw.write("%d %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f" % (0, proj_2d_gt[0,0], proj_2d_gt[1,0], proj_2d_gt[0,1], proj_2d_gt[1,1], proj_2d_gt[0,2], proj_2d_gt[1,2], proj_2d_gt[0,3], proj_2d_gt[1,3], proj_2d_gt[0,4], proj_2d_gt[1,4], proj_2d_gt[0,5], proj_2d_gt[1,5], proj_2d_gt[0,6], proj_2d_gt[1,6], proj_2d_gt[0,7], proj_2d_gt[1,7], proj_2d_gt[0,8], proj_2d_gt[1,8], xrange/width, yrange/height))
	idx+=1

for filename in os.listdir(scene_dir):
	if filename.endswith("_category.png") or filename.endswith("_depth.png") or filename.endswith("_kinect.png"):
		os.remove(scene_dir+filename)
	elif filename.endswith("_color.jpg"):
		if not os.path.exists(scene_dir+"JPEGImages"):
			os.mkdir(scene_dir+"JPEGImages")
		shutil.move(scene_dir+filename, scene_dir+"JPEGImages/")



