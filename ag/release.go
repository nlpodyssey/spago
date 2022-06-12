// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

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
func ReleaseGraph(nodes ...Node) {
	for _, node := range nodes {
		if op, ok := node.(*Operator); ok && op.function != nil {
			ReleaseGraph(op.function.Operands()...)
			op.releaseValue()
			op.ZeroGrad()
			op.function = nil
			op.cond.L = nil
		}
	}
}
