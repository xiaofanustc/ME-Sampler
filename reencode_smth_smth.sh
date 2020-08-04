#!/usr/bin/env bash

if [ "$#" -ne 2 ]; then
    echo "Usage: ./reencode.sh [input dir] [output dir]"
fi

indir=$1
outdir=$2

mkdir outdir
if [[ ! -d "${outdir}" ]]; then
  echo "${outdir} doesn't exist. Creating it.";
  mkdir -p ${outdir}
fi

MY_SAVEIFS=$IFS
IFS=$'\n'

for c in $(ls ${indir})
do
	for inname in $(ls ${indir}/${c}/*mp4)
	do
		class_path="$(dirname "$inname")"
        echo $class_path
		
        class_name="${class_path##*/}"
        echo $class_name
        echo $inname

		outname="${outdir}/${class_name}/${inname##*/}"
		outname="${outname%.*}.mp4"

		mkdir -p "$(dirname "$outname")"
        #ffmpeg -i ${inname} -vf scale=340:256,setsar=1:1 -q:v 1 -c:v mpeg4 -f rawvideo ${outname}
        ffmpeg -i ${inname} -q:v 1 -c:v mpeg4 -f rawvideo ${outname}
	done
done

