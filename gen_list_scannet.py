import os
from shutil import copyfile
import shutil

# base_folder = "Dataset/scannet/scans/"
base_folder = "/nfs/nas12.ethz.ch/fs1201/infk_ivc_students/cvg-students/zhoum/scannet/scans/"

target_name = "_couch"
fw_train = open(base_folder+"train"+target_name+".txt", "w+")
fw_test = open(base_folder+"test"+target_name+".txt", "w+")

total_count = 0
for scene in os.listdir(base_folder):
	label_list = []
	if "scene" in scene:
		if os.path.exists(base_folder+scene+"/labels"+target_name):
			shutil.rmtree(base_folder+scene+"/labels"+target_name)
		
		for label in os.listdir(base_folder+scene):
			if label.startswith("labels") and label.endswith(target_name):
				label_list.append(label)
		for label in label_list:
			if not os.path.exists(base_folder+scene+"/labels"+target_name):
				os.makedirs(base_folder+scene+"/labels"+target_name)
			print (scene+'/'+label)
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
							total_count+=1
							copyfile(base_folder+scene+"/"+label+"/"+txt, base_folder+scene+"/labels"+target_name+"/"+txt)

							#fw_train.write("../"+base_folder+scene+"/JPEGImages/"+txt.replace("txt", "jpg")+"\n")
							#fw_test.write("../"+base_folder+scene+"/JPEGImages/"+txt.replace("txt", "jpg")+"\n")
							fw_train.write("../../../../nfs/nas12.ethz.ch/fs1201/infk_ivc_students/cvg-students/zhoum/scannet/scans/"+scene+"/JPEGImages/"+txt.replace("txt", "jpg")+"\n")
							fw_test.write("../../../../nfs/nas12.ethz.ch/fs1201/infk_ivc_students/cvg-students/zhoum/scannet/scans/"+scene+"/JPEGImages/"+txt.replace("txt", "jpg")+"\n")


print (total_count)
