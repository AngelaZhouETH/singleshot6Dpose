import os
from shutil import copyfile
import shutil

# base_folder = "Dataset/scannet/scans/"
base_folder = "/nfs/nas12.ethz.ch/fs1201/infk_ivc_students/cvg-students/zhoum/scannet/scans/"
fw_train = open(base_folder+"train.txt", "w+")
fw_test = open(base_folder+"test.txt", "w+")

total_count = 0
for scene in os.listdir(base_folder):
	label_list = []
	if "scene" in scene:
		for label in os.listdir(base_folder+scene):
			if label.startswith("labels") and label.endswith("_chair"):
				label_list.append(label)
		
		if os.path.exists(base_folder+scene+"/labels"):
			shutil.rmtree(base_folder+scene+"/labels")
		for label in label_list:
			if not os.path.exists(base_folder+scene+"/labels"):
				os.makedirs(base_folder+scene+"/labels")
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
							copyfile(base_folder+scene+"/"+label+"/"+txt, base_folder+scene+"/labels/"+txt)

							#fw_train.write("../"+base_folder+scene+"/JPEGImages/"+txt.replace("txt", "jpg")+"\n")
							#fw_test.write("../"+base_folder+scene+"/JPEGImages/"+txt.replace("txt", "jpg")+"\n")
							fw_train.write("/nfs/nas12.ethz.ch/fs1201/infk_ivc_students/cvg-students/zhoum/scannet/scans/"+scene+"/JPEGImages/"+txt.replace("txt", "jpg")+"\n")
							fw_test.write("/nfs/nas12.ethz.ch/fs1201/infk_ivc_students/cvg-students/zhoum/scannet/scans/"+scene+"/JPEGImages/"+txt.replace("txt", "jpg")+"\n")


print (total_count)
