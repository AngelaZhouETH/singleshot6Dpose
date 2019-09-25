for dir in "."/*; do
	if [[ $dir == *"scene"* ]]; then
		name=$(echo $dir | cut -d'/' -f 2)
		python reader.py --filename $dir'/'$name'.sens' --output_path $dir --export_color_images --export_poses --export_intrinsics
	fi
done