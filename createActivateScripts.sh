#!/bin/bash

# Set the target directory where the modified activate scripts will be placed
target_dir="/opt/python"

# Iterate over the directories under /opt/python starting with "cp"
for dir in $target_dir/cp*; do
    if [ -d "$dir" ]; then
        # Create the bin directory if it doesn't exist
        mkdir -p "$dir/bin"

        # Generate the modified activate script
        sed -e "s|/venv36|$dir|g" -e "s|(venv36) |(`basename "$dir"`) |g" /venv36/bin/activate > "$dir/bin/activate"

        # Set execute permissions for the activate script
        chmod +x "$dir/bin/activate"

        echo "Modified activate script created: $dir/bin/activate"
    fi
done

