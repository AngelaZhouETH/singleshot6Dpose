for dir in "."/*; do
	if [[ $dir == *"scene"* ]]; then
		name=$(echo $dir | cut -d'/' -f 2)
		mv $dir'/color' $dir'/JPEGImages'
		#unzip $dir'/'$name'_2d-instance.zip' -d $dir
	fi
done
