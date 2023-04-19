#!/bin/bash
dev=$1
ini=$2
fin=$3


for ((i=$ini;i<=$fin;i++))
do
    echo "$i"
    python execute_gender_classification.py -n $i -d $dev
    echo ""
    echo ""
done