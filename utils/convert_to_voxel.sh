#!/bin/bash

for filename in *.stl; do
./binvox -pb -c -d 64 "$filename"
done
