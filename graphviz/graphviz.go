// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package graphviz

import (
	"os"

	"github.com/awalterschulze/gographviz"
	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
)

// Marshal the graph in a dot (graphviz).
func Marshal[T mat.DType](g *ag.Graph[T]) ([]byte, error) {
	gv, err := BuildGraph(g, Options{ColoredTimeSteps: true})
	if err != nil {
		return nil, err
	}
	return []byte(gv.String()), nil
}

// BuildGraph creates a gographviz representation of the Graph.
func BuildGraph[T mat.DType](g *ag.Graph[T], options Options) (gographviz.Interface, error) {
	return newBuilder(g, options).build()
}

// Save saves a gographviz graph to a DOT file.
func Save(gv gographviz.Interface, filename string) (err error) {
	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer func() {
		if e := f.Close(); e != nil && err == nil {
			err = e
		}
	}()
	_, err = f.WriteString(gv.String())
	if err != nil {
		return err
	}
	return nil
}
