#!/bin/bash

for filename in *.stl; do
./binvox -pb -c -d $1 "$filename"
./binvox -pb -c -rotx -d $1 "$filename"
./binvox -pb -c -rotx -rotx -d $1 "$filename"
./binvox -pb -c -rotx -rotx -rotx -d $1 "$filename"
./binvox -pb -c -rotz -rotx -d $1 "$filename"
./binvox -pb -c -rotz -rotz -rotz -rotx -d $1 "$filename"
done
