#!/bin/bash
echo $1 $2
folder=$1
number=$2

if [ ! $number -ne $number ] 2>/dev/null; then
    echo chosen $number best tires
else
    number=6
    echo chosen $number best tries
fi
declare -a names
if [ -d$folder ]; then
    imagename=$folder #(`echo $folder | sed 's/.*_\\\*//g'`)
    imagename=best_graphics_cup
    echo $imagename
    newfolder=best_$folder
    cd $folder
    names=(`ls [0-9]_*.png | sed 's/\-/\ /g' | sort -k 2 | sed 's/\ /\-/g' | head -$number`)
    echo $names
    cd ..
    if [ -d $newfolder ]; then
        rm -r $newfolder
        echo $newfolder "removed"
    fi
    mkdir $newfolder
    cd $folder
    for i in ${names[@]}; do
        cp ${i} ../$newfolder/${i}
        echo "${i} copied in ../$newfolder/${i}"
    done
    cd ../$newfolder
    montage -geometry -39+0 ${names[*]} $imagename.png
    eog $imagename.png &
fi

exit 0

