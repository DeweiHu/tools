#!/usr/bin/env bash

Dir=/home/dewei/Desktop/octa/temp
greedy=/home/dewei/tool/oguzi/bin/greedy

im_fix=$Dir/im_fix.nii
im_mov=$Dir/im_mov.nii
warp=$Dir/warp.nii
im_warped=$Dir/warped.nii

$greedy -d 2 -i $im_fix $im_mov -o $warp -n 100x50x10 -m NCC 4x4 -threads 1
$greedy -d 2 -r $warp -rf $im_fix -rm $im_mov $im_warped
