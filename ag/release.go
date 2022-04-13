// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	"github.com/nlpodyssey/spago/mat"
)

// ReleaseGraph traverses the (sub-)graphs consisting of operators and
// nested operands, starting from the given nodes, and frees the resources
// of each operator.
//
// Any Node implementation can be passed to the function, however only Operators
// and their operands will be taken into account, and the rest simply ignored.
//
// This function is not concurrency safe.
//
// Freed resources include, but are not limited to, the value and the gradients.
// Any freed operator MUST not be used after this operation is performed.
func ReleaseGraph[T mat.DType](nodes ...Node[T]) {
	visited := make(map[*Operator[T]]struct{})
	toVisit := make([]*Operator[T], 0, len(nodes))

	for _, node := range nodes {
		if op, ok := node.(*Operator[T]); ok {
			toVisit = append(toVisit, op)
		}
	}

	for len(toVisit) > 0 {
		lastIndex := len(toVisit) - 1
		op := toVisit[lastIndex]
		toVisit[lastIndex] = nil
		toVisit = toVisit[:lastIndex]

		if _, ok := visited[op]; ok {
			continue
		}
		visited[op] = struct{}{}

		for _, operand := range op.function.Operands() {
			if oo, ok := operand.(*Operator[T]); ok {
				toVisit = append(toVisit, oo)
			}
		}

		op.releaseValue()
		op.ZeroGrad()
		op.function = nil
		op.cond.L = nil
	}
}
