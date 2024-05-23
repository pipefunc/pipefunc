# This file is part of the pipefunc repository: .github/add_filename_header.sh
#!/bin/sh

modified=$(git diff --cached --name-only --diff-filter=ACMR)
echo "Modified files: $modified"

[ -z "$modified" ] && exit 0

for file in $modified; do
    echo "Processing file: $file"
    if [ -f "$file" ]; then
        echo "$file is a regular file."
        # Check if the header is already in the file
        if ! grep -q "^# This file is part of the pipefunc repository: $file" "$file"; then
            echo "Adding header to $file"
            # Add the header to the top of the file
            {
                echo "# This file is part of the pipefunc repository: $file"
                cat "$file"
            } > "$file.new"
            mv "$file.new" "$file"
            git add "$file"
        fi
    fi
done
