// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

// ReleaseGraphFunc is returned by the Backward function.
type ReleaseGraphFunc func()

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
func ReleaseGraph(nodes ...DualValue) {
	for _, node := range nodes {
		if op, ok := node.(*Operator); ok && op.fn != nil {
			ReleaseGraph(op.Operands()...)
			op.ZeroGrad()
			op.releaseValue()
			op.fn = nil
		}
	}
}
