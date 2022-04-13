// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import "github.com/nlpodyssey/spago/mat"

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
	for _, node := range nodes {
		if op, ok := node.(*Operator[T]); ok {
			releaseGraph[T](visited, op)
		}
	}
}

func releaseGraph[T mat.DType](visited map[*Operator[T]]struct{}, op *Operator[T]) {
	if _, ok := visited[op]; ok {
		return
	}
	visited[op] = struct{}{}

	op.releaseValue()
	op.ZeroGrad()

	for _, operand := range op.function.Operands() {
		if oo, ok := operand.(*Operator[T]); ok {
			releaseGraph[T](visited, oo)
		}
	}

	op.function = nil
	op.valueCond.L = nil
	op.gradCond.L = nil
}
