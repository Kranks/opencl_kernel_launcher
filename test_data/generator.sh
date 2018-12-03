#!/bin/bash
# generateur de matrice d'adjacente pour un reseau circulaire
echo $1 > mat
for i in $(seq 0 1 $(($1 - 1)))
do
    for j in $(seq 0 1 $(($1 - 1)))
    do
        if [ $j -eq $i ]
        then
            echo -n "0 " >> mat
        elif [ $j -eq $(($i + 1)) ]
        then 
            echo -n "1 " >> mat
        elif [ $i -eq $(($1 - 1)) -a $j -eq 0 ]
        then
            echo -n "1 " >> mat
        else
            echo -n "$(($1 + 1)) " >> mat
        fi
    done
    echo " " >> mat
done
