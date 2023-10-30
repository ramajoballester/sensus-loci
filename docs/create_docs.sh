#!/bin/bash

rm -rf ./docs/build

# Define input and output file paths
input_file="./README.md"
output_file="./docs/source/readme.md"

# Use grep to filter link lines and sed to remove them
grep -v '^![^)]*)' "$input_file" > "$output_file"

echo "Filtered $input_file saved to $output_file"

# Build the docs locally
sphinx-build -M html docs/source/ docs/build/ -a