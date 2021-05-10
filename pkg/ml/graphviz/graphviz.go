// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package graphviz

import (
	"github.com/awalterschulze/gographviz"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
)

// BuildGraph creates a gographviz representation of the Graph.
func BuildGraph(g *ag.Graph, options Options) (gographviz.Interface, error) {
	return newBuilder(g, options).build()
}
