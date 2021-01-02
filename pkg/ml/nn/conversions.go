// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nn

import (
	"fmt"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
)

// ToNodes coerces the given value to a slice of nodes []ag.Node.
//
// If the value is already of type []ag.Node, it is simply returned unmodified.
// If the value is a single ag.Node, the method returns a new slice of nodes, containing just
// that node. In case of any other type, the method panics.
func ToNodes(i interface{}) []ag.Node {
	switch ii := i.(type) {
	case []ag.Node:
		return ii
	case ag.Node:
		return []ag.Node{ii}
	default:
		panic(fmt.Errorf("nn: cannot coerce to []ag.Node a value of type %T", ii))
	}
}

// ToNode coerces the given value to an ag.Node.
//
// If the underlying value is already of type ag.Node, it is simply returned unmodified.
// If the value is a slice of nodes []ag.Node which contains exactly one item, the method returns
// just that node. If the value is a slice of zero, two or more nodes, the method panics.
// In case of any other type, the method panics as well.
func ToNode(i interface{}) ag.Node {
	switch ii := i.(type) {
	case ag.Node:
		return ii
	case []ag.Node:
		if len(ii) != 1 {
			panic(fmt.Errorf("nn: cannot coerce to ag.Node a slice of nodes with len %d", len(ii)))
		}
		return ii[0]
	default:
		panic(fmt.Errorf("nn: cannot coerce to ag.Node a value of type %T", ii))
	}
}
