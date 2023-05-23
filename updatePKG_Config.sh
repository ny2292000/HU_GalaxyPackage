update_pkg_config_path() {
    # Search for all directories containing .pc files
    dirs="$(find / -type f -name '*.pc' -exec dirname {} \; | sort -u)"

    # Add the directories to PKG_CONFIG_PATH, eliminating duplicates
    export PKG_CONFIG_PATH="$(echo -n "$PKG_CONFIG_PATH:$dirs" | tr '\n' ':' | sed 's/:*$//')"
    export PKG_CONFIG_PATH="$(perl -e 'print join(":", grep { not $seen{$_}++ } split(/:/, $ENV{PKG_CONFIG_PATH}))')"
}

 update_pkg_config_path
 echo $PKG_CONFIG_PATH