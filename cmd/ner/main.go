// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This is the first attempt to launch a sequence labeling server from the command line.
// Please note that configurations, parameter loading, and who knows how many other things, require heavy refactoring!
package main

import (
	"os"

	"github.com/nlpodyssey/spago/cmd/ner/internal/app"
)

func main() {
	app.NewNERApp().Run(os.Args)
}
