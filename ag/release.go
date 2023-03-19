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
func ReleaseGraph(nodes ...Node) {
	for _, node := range nodes {
		if op, ok := node.(*Operator); ok && op.backwardPass != nil {
			ReleaseGraph(op.Operands()...)
			op.releaseValue()
			op.ZeroGrad()
			op.backwardPass = nil
			op.cond.L = nil
		}
	}
}

// NodesTracker is a helper struct that can be used to track nodes and release them
type NodesTracker struct {
	nodes []Node
}

// TrackNode adds the given node to the list of nodes to be released
func (nt *NodesTracker) TrackNode(node Node) Node {
	nt.nodes = append(nt.nodes, node)
	return node
}

// TrackNodes adds the given nodes to the list of nodes to be released
func (nt *NodesTracker) TrackNodes(nodes ...Node) []Node {
	nt.nodes = append(nt.nodes, nodes...)
	return nodes
}

// ReleaseNodes releases all the nodes tracked by this tracker
func (nt *NodesTracker) ReleaseNodes() {
	ReleaseGraph(nt.nodes...)
	nt.nodes = nil
}
