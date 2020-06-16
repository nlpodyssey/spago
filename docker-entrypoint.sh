#!/usr/bin/env sh

# Copyright 2020 spaGO Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# The script docker-entrypoint.sh wraps access to the demo
# programs for named entities recognition (ner-server), model
# importing (hugging_face_importer), and question answering
# (bert_server).

set -o errexit
set -o pipefail
set -o nounset
# set -o xtrace

# The help screen is printed to the user when no commands
# are given, or when the command "help" is given.
if [[ -z "$1" -o "$1" = "help" ]]; then
    echo "spaGO is a beautiful and maintainable machine learning library written in Go designed to support relevant neural network architectures in natural language processing tasks."
    echo ""
    echo "Usage:"
    echo "  <command> [arguments]"
    echo ""
    echo "The commands are:"
    echo ""
    echo "   bert_server             demo server for question answering"
    echo "   hugging_face_importer   demo program for model importing"
    echo "   ner-server              demo server for named entities recognition"
    echo ""
    echo "See README.md for more information about run the demo servers using docker."

    exit 0
fi

# Run the commands defined by the Dockerfile CMD directive.
echo "Running command '$@'"
exec "$@"
