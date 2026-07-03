#!/bin/bash

# Current directory
FOLDER="$(pwd)"

# Output file
OUTPUT_FILE="$FOLDER/name.txt"

# Empty the output file if it already exists
> "$OUTPUT_FILE"

# List all files (excluding name.txt), sort them, and prefix each with "* "
find "$FOLDER" -maxdepth 1 -type f ! -name "name.txt" -exec basename {} \; | sort | while read -r file
do
    echo "* $file" >> "$OUTPUT_FILE"
done

echo "Done! File names have been written to $OUTPUT_FILE"