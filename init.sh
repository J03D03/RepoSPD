#!/bin/bash

# Initialize RepoSPD container
echo "Initializing RepoSPD container..."

# Set executable permissions for joern
chmod +x -R /RepoSPD/joern

# Fix EMPTY JAR
echo "Fetching Joern libraries..."
cd /RepoSPD/joern/lib && ./fetch-joern-libs.sh

# Set data paths in Python files
echo "Updating data paths..."
sed -i "s|root = '/data1/lzr/code/GraphTwin9/release'|root = '/RepoSPD'|g" /RepoSPD/data_preproc/data_loader.py 2>/dev/null || true
sed -i "s|root = '/data1/lzr/code/GraphTwin9/release'|root = '/RepoSPD'|g" /RepoSPD/add_dependency/add_dep.py 2>/dev/null || true
sed -i "s|root = '/data1/lzr/code/GraphTwin9/release'|root = '/RepoSPD'|g" /RepoSPD/add_dependency/get_depend.py 2>/dev/null || true

if [ $# -eq 0 ]; then
    exec /bin/bash
else
    exec "$@"
fi
