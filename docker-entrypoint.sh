#!/usr/bin/env sh

# Copyright 2020 spaGO Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# docker-entrypoint.sh wraps access to the demo programs for
# named entities recognition (ner-server), model importing
# (hugging_face_importer), and question answering (bert_server).

set -o errexit
set -o pipefail
set -o nounset
# set -o xtrace

if [[ -z "$1" -o "$1" = "help" ]]; then
    echo "spaGO is a beautiful and maintainable machine learning library written in Go designed to support relevant neural network architectures in natural language processing tasks."
    echo ""
    echo "Usage:"
    echo "  <command> [arguments]"
    echo ""
    echo "The commands are:"
    echo ""
    echo "   ner-server              demo for named entities recognition"
    echo "   hugging_face_importer   demo for model importing"
    echo "   bert_server             demo for question answering"
    echo ""
    echo "The models folder should be bind-mounted like this."
    echo ""
    echo "docker run --rm -it -p:1987:1987 -v ~/.spago:/tmp/spago spago:main ./ner-server 1987 /tmp/spago goflair-en-ner-fast-conll03"
    echo ""
    echo "See README.md for more information about using docker to run the demos."

    exit 0
fi

# needed to run parameters CMD
echo "Running command '$@'"
exec "$@"
