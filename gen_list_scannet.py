import os
from shutil import copyfile
import shutil
from plyfile import PlyData, PlyElement
import numpy as np


# base_folder = "Dataset/scannet/scans/"
base_folder = "/nfs/nas12.ethz.ch/fs1201/infk_ivc_students/cvg-students/zhoum/scannet/scans/"

target_name = "_sofa"
upper_bound = 10
lower_bound = 0
if target_name=="_table":
	base_mesh = base_folder+"scene0001_01/new_mesh7.ply"
elif target_name=="_couch":
	base_mesh=base_folder+"scene0001_00/new_mesh1.ply"
elif target_name=="_chair":
	base_mesh=base_folder+"scene0002_00/new_mesh18.ply"
elif target_name=="_sofa":
        base_mesh=base_folder+"scene0001_00/new_mesh1.ply"

plydata = PlyData.read(base_mesh)
base_mesh_verts = np.zeros(shape=[plydata['vertex'].count, 3], dtype=np.float32)
base_mesh_verts[:,0] = plydata['vertex']['x']
base_mesh_verts[:,1] = plydata['vertex']['y']
base_mesh_verts[:,2] = plydata['vertex']['z']
min_x = np.min(base_mesh_verts[:,0])
max_x = np.max(base_mesh_verts[:,0])
min_y = np.min(base_mesh_verts[:,1])
max_y = np.max(base_mesh_verts[:,1])
min_z = np.min(base_mesh_verts[:,2])
max_z = np.max(base_mesh_verts[:,2])
base_bbox_x = max_x - min_x
base_bbox_y = max_y - min_y
base_bbox_z = max_z - min_z


fw_train = open(base_folder+"train"+target_name+".txt", "w+")
fw_test = open(base_folder+"test"+target_name+".txt", "w+")

total_imgs_count = 0
train_imgs_count = 0
test_imgs_count = 0
total_obj_count = 0
filt_obj_count = 0
train_list = []
test_list = []
for scene in os.listdir(base_folder):
	label_list = []
	if "scene" in scene:
		scene_id = scene.split("_")[0][5:]
		if os.path.exists(base_folder+scene+"/labels"+target_name):
			shutil.rmtree(base_folder+scene+"/labels"+target_name)
		
		for label in os.listdir(base_folder+scene):
			if label.startswith("newlabels") and label.endswith(target_name):
				label_list.append(label)
		total_obj_count += len(label_list)
		for label in label_list:
			if not os.path.exists(base_folder+scene+"/labels"+target_name):
				os.makedirs(base_folder+scene+"/labels"+target_name)
			print (scene+'/'+label)
			mesh_obj = base_folder+scene+'/new_mesh'+label.split('_')[0][9:]+'.ply'
			plydata = PlyData.read(mesh_obj)
			mesh_obj_verts = np.zeros(shape=[plydata['vertex'].count, 3], dtype=np.float32)
			mesh_obj_verts[:,0] = plydata['vertex']['x']
			mesh_obj_verts[:,1] = plydata['vertex']['y']
			mesh_obj_verts[:,2] = plydata['vertex']['z']
			min_x = np.min(mesh_obj_verts[:,0])
			max_x = np.max(mesh_obj_verts[:,0])
			min_y = np.min(mesh_obj_verts[:,1])
			max_y = np.max(mesh_obj_verts[:,1])
			min_z = np.min(mesh_obj_verts[:,2])
			max_z = np.max(mesh_obj_verts[:,2])
			mesh_bbox_x = max_x - min_x
			mesh_bbox_y = max_y - min_y
			mesh_bbox_z = max_z - min_z
			if (mesh_bbox_x/base_bbox_x > upper_bound or mesh_bbox_x/base_bbox_x < lower_bound or mesh_bbox_y/base_bbox_y > upper_bound or mesh_bbox_y/base_bbox_y < lower_bound or mesh_bbox_z/base_bbox_z > upper_bound or mesh_bbox_z/base_bbox_z < lower_bound):
				filt_obj_count += 1
				continue

			prev_idx = -10
			for txt in os.listdir(base_folder+scene+"/"+label):
				if txt.endswith("txt"):
					flag = True
					for other in label_list:
						if label != other and os.path.isfile(base_folder+scene+"/"+other+"/"+txt):
							flag = False
							break
					if (flag):
						idx = int(txt.replace(".txt",""))
						if idx - prev_idx > 4:
							prev_idx = idx
							total_imgs_count+=1
							copyfile(base_folder+scene+"/"+label+"/"+txt, base_folder+scene+"/labels"+target_name+"/"+txt)

							if (scene_id in train_list):
								#fw_train.write("../"+base_folder+scene+"/JPEGImages/"+txt.replace("txt", "jpg")+"\n")
								fw_train.write("../../../../nfs/nas12.ethz.ch/fs1201/infk_ivc_students/cvg-students/zhoum/scannet/scans/"+scene+"/JPEGImages/"+txt.replace("txt", "jpg")+"\n")
								train_imgs_count += 1
							elif (scene_id in test_list):
								#fw_test.write("../"+base_folder+scene+"/JPEGImages/"+txt.replace("txt", "jpg")+"\n")
								fw_test.write("../../../../nfs/nas12.ethz.ch/fs1201/infk_ivc_students/cvg-students/zhoum/scannet/scans/"+scene+"/JPEGImages/"+txt.replace("txt", "jpg")+"\n")
								test_imgs_count += 1
							elif (train_imgs_count > test_imgs_count*2):
								#fw_test.write("../"+base_folder+scene+"/JPEGImages/"+txt.replace("txt", "jpg")+"\n")
								fw_test.write("../../../../nfs/nas12.ethz.ch/fs1201/infk_ivc_students/cvg-students/zhoum/scannet/scans/"+scene+"/JPEGImages/"+txt.replace("txt", "jpg")+"\n")
								test_imgs_count += 1
								test_list.append(scene_id)
							else:
								#fw_train.write("../"+base_folder+scene+"/JPEGImages/"+txt.replace("txt", "jpg")+"\n")
								fw_train.write("../../../../nfs/nas12.ethz.ch/fs1201/infk_ivc_students/cvg-students/zhoum/scannet/scans/"+scene+"/JPEGImages/"+txt.replace("txt", "jpg")+"\n")
								train_imgs_count += 1
								train_list.append(scene_id)
print ("total_imgs_count:")
print (total_imgs_count)
print ("train_imgs_count:")
print (train_imgs_count)
print ("test_imgs_count:")
print (test_imgs_count)
print ("total_obj_count:")
print (total_obj_count)
print ("filt_obj_count:")
print (filt_obj_count)
print ("train scenes count:")
print (len(train_list))
print ("test scenes count:")
print (len(test_list))
