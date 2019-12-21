// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	"sync"
)

type Graph struct {
	// to avoid data race during concurrent computations
	mu sync.Mutex
	// maxId is the id of the last inserted node (corresponds of len(nodes)-1)
	maxId int64
	// the maximum depth reached by a node of the graph
	maxDepth int
	// nodes contains the list of nodes of the graph. The indices of the list are the nodes ids.
	nodes []*nodeInfo
}

type nodeInfo struct {
	node Node
	// depth is the maximum depth reached by the node/.
	depth int
	// descendants contains the ids of all descendants including the node itself.
	descendants []int64
}

// NewGraph returns a new initialized graph.
func NewGraph() *Graph {
	return &Graph{
		maxId:    0,
		maxDepth: 0,
		nodes:    make([]*nodeInfo, 0),
	}
}

func (g *Graph) Reset() {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.maxId = 0
	g.maxDepth = 0
	g.nodes = make([]*nodeInfo, 0)
}
