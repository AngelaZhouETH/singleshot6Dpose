for dir in "."/*; do
	if [[ $dir == *"scene"* ]]; then
		name=$(echo $dir | cut -d'/' -f 2)
		#if [ -d $dir'/pose' ]; then
		#	rm $dir'/'$name'.sens'
		#	zip -r $dir'.zip' $dir
		#fi
		#if [ -f $dir'.zip' ]; then
		#	rm -r $dir
		#fi
		if [ ! -f $dir'.zip' ]; then
			if [ -f $dir'/'$name'.sens' ]; then
				python reader.py --filename $dir'/'$name'.sens' --output_path $dir --export_color_images --export_poses --export_intrinsics
			fi
		fi
	fi
done
