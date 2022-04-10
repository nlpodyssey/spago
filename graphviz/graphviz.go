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

// Graphviz creates a gographviz representation of the (sub-)graph up to outputNodes .
func Graphviz[T mat.DType](outputNodes []ag.Node[T], options ...Option[T]) (gographviz.Interface, error) {
	return newBuilder(outputNodes, options...).build()
}

// Marshal the (sub-)graph up to outputNodes into DOT (graphviz) format.
func Marshal[T mat.DType](outputNodes []ag.Node[T]) ([]byte, error) {
	gv, err := Graphviz(outputNodes, WithColoredTimeSteps[T](true))
	if err != nil {
		return nil, err
	}
	return []byte(gv.String()), nil
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
