#!/bin/bash
dev=$1
ini=$2
fin=$3


for ((i=$ini;i<=$fin;i++))
do
    echo "$i"
    python execute_classification_training.py -n $i -d $dev
    echo ""
    echo ""
done
