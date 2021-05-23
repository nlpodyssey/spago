// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package dot creates a graphviz compatible version of the ag.Graph.
package dot

import (
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/graphviz"
)

// Marshal the graph in a dot (graphviz).
func Marshal(g *ag.Graph) ([]byte, error) {
	gv, err := graphviz.BuildGraph(g, graphviz.Options{ColoredTimeSteps: true})
	if err != nil {
		return nil, err
	}
	return []byte(gv.String()), nil
}
