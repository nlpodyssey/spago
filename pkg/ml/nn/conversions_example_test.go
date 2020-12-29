// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nn_test

import (
	"fmt"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
)

func ExampleToNode() {
	g := ag.NewGraph()

	a := g.NewScalar(1)
	b := []ag.Node{g.NewScalar(2)}

	fmt.Println(nn.ToNode(a).Value())
	fmt.Println(nn.ToNode(b).Value())

	// Output:
	// [1]
	// [2]
}

func ExampleToNodes() {
	g := ag.NewGraph()

	a := g.NewScalar(1)
	b := []ag.Node{g.NewScalar(2), g.NewScalar(3)}

	fmt.Printf("len: %d, first value: %v\n", len(nn.ToNodes(a)), nn.ToNodes(a)[0].Value())
	fmt.Printf("len: %d, first value: %v\n", len(nn.ToNodes(b)), nn.ToNodes(b)[0].Value())

	// Output:
	// len: 1, first value: [1]
	// len: 2, first value: [2]
}
