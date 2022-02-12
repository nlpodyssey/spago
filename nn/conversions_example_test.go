// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nn_test

import (
	"fmt"
	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/nn"
)

func ExampleToNode() {
	g := ag.NewGraph[float32]()

	a := g.NewScalar(1)
	b := []ag.Node[float32]{g.NewScalar(2)}

	fmt.Println(nn.ToNode[float32](a).Value())
	fmt.Println(nn.ToNode[float32](b).Value())

	// Output:
	// [1]
	// [2]
}

func ExampleToNodes() {
	g := ag.NewGraph[float32]()

	a := g.NewScalar(1)
	b := []ag.Node[float32]{g.NewScalar(2), g.NewScalar(3)}

	fmt.Printf("len: %d, first value: %v\n", len(nn.ToNodes[float32](a)), nn.ToNodes[float32](a)[0].Value())
	fmt.Printf("len: %d, first value: %v\n", len(nn.ToNodes[float32](b)), nn.ToNodes[float32](b)[0].Value())

	// Output:
	// len: 1, first value: [1]
	// len: 2, first value: [2]
}
