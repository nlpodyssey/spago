#!/usr/bin/env bash

# Copyright 2021 spaGO Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

set -e

script_name=$(basename "$0")
usage="
DEPRECATED: this is a deprecated utility, temporary useful only while
developing the adoption of generics.

This is a small utility script to modify all source code files of spaGO in
order to use the desired floating point precision. It simply modifies
the alias value of the type \"mat.Float\", changing it to either \"float32\"
or \"float64\".

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
    from_bits=64
    to_bits=32
    ;;
  64)
    from_bits=32
    to_bits=64
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

sed -i "s/type Float = float$from_bits/type Float = float$to_bits/g" ./pkg/mat/float.go
