#!/usr/bin/env bash

# Copyright 2021 spaGO Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

set -e

script_name=$(basename "$0")
usage="
This is a small utility script to modify all source code files of spaGO in
order to use the desired floating point precision. It simply modifies import
statements, choosing between the packages \"mat32\" or \"mat64\" (including
nested packages). Since \"mat32\" and \"mat64\" are always aliased as \"mat\",
and \"mat.Float\" is always preferred over explicit float32 or float64 types,
there is no need to change anything else.

Usage: $script_name [OPTION]

Options:
  -h, -help, --help    Show usage and exit
  32                   Change the codebase using float32 as main type
                       (mat32 package)
  64                   Change the codebase using float64 as main type
                       (mat64 package)
"

if [[ $# -ne 1 ]]; then
  echo "Wrong number of parameters."
  echo "$usage"
  exit 1
fi

case $1 in
  32)
    from_package=mat64
    to_package=mat32
    ;;
  64)
    from_package=mat32
    to_package=mat64
    ;;
  -h | -help | --help)
    echo "$usage"
    exit
    ;;
  *)
    echo "Wrong option."
    echo "$usage"
    exit 1
    ;;
esac

script_path=$(dirname "$0")
find "$script_path" \( -type d -name "mat??" -prune \) -name "*.go" -o -type f -print0 \
  | xargs -0 sed -i "s/\"github.com\/nlpodyssey\/spago\/pkg\/$from_package/\"github.com\/nlpodyssey\/spago\/pkg\/$to_package/g"
