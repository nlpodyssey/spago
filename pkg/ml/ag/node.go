// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

type Node interface {
	GradValue
	// Graph returns the graph this node belongs to.
	Graph() *Graph
	// Id returns the id of the node in the graph.
	Id() int64
}
