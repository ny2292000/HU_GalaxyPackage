find_directories() {
    local scope_string=${1:-"/"}
    local search_string=$2
    local temp_file=$(mktemp)
    find "$scope_string" -type d -name "*$search_string*" 2>/dev/null | awk -F/ '/boost/{sub(/boost.*/, "boost"); print}' | sort -u > "$temp_file"

    local previous_line=""
    local current_line=""
    local first_line=true

    while IFS= read -r line; do
        current_line="$line"
        if [[ "$first_line" == true ]]; then
            echo "$line"
            first_line=false
        elif [[ ! "$current_line" == "$previous_line"* ]]; then
            echo "$line"
        fi
        previous_line="$current_line"
    done < "$temp_file"

    rm -f "$temp_file"
}
