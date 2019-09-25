import os
from shutil import copyfile
import shutil

base_folder = "Data_raw/house_test/"
to_folder = "Data_raw/house_single_test/"
fw_train = open(to_folder+"train.txt", "w+")
fw_test = open(to_folder+"test.txt", "w+")

total_count = 0
for name in os.listdir(base_folder):
	idx_count = 0
	if not name.startswith(".") and not name.endswith("txt"):
		for label in os.listdir(base_folder+name):
			if label.startswith("label"):
				idx_count += 1
		
		for i in range(idx_count):
			if not os.path.exists(to_folder+name+"/labels"):
				os.makedirs(to_folder+name+"/labels")
			if not os.path.exists(to_folder+name+"/JPEGImages"):
				os.makedirs(to_folder+name+"/JPEGImages")
			for txt in os.listdir(base_folder+name+"/labels"+str(i)):
				if txt.endswith("txt"):
					flag = True
					for j in range(idx_count):
						if i != j and os.path.isfile(base_folder+name+"/labels"+str(j)+"/"+txt):
							flag = False
							break
					if (flag):
						total_count+=1
						copyfile(base_folder+name+"/labels"+str(i)+"/"+txt, to_folder+name+"/labels/"+txt)
						# copyfile(base_folder+name+"/outputcamerafile", to_folder+name+"/outputcamerafile")
						# copyfile(base_folder+name+"/house.json", to_folder+name+"/house.json")
						# copyfile(base_folder+name+"/house.mtl", to_folder+name+"/house.mtl")
						# copyfile(base_folder+name+"/house.obj", to_folder+name+"/house.obj")
						copyfile(base_folder+name+"/JPEGImages/"+txt.replace("txt", "jpg"), to_folder+name+"/JPEGImages/"+txt.replace("txt", "jpg"))
						fw_train.write("../"+to_folder+name+"/JPEGImages/"+txt.replace("txt", "jpg")+"\n")
						fw_test.write("../"+to_folder+name+"/JPEGImages/"+txt.replace("txt", "jpg")+"\n")

print (total_count)