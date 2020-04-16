// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import "github.com/nlpodyssey/spago/pkg/mat"

type wrapper struct {
	GradValue
	graph    *Graph
	id       int64
	wrapGrad bool
}

// Id returns the id of the node in the graph.
func (r *wrapper) Id() int64 {
	return r.id
}

// Graph returns the graph this node belongs to.
func (r *wrapper) Graph() *Graph {
	return r.graph
}

// Grad returns the gradients accumulated during the backward pass.
func (r *wrapper) Grad() mat.Matrix {
	if r.wrapGrad {
		return r.GradValue.Grad()
	} else {
		return nil
	}
}

// PropagateGrad propagates the gradients to the node.
func (r *wrapper) PropagateGrad(gx mat.Matrix) {
	if r.wrapGrad {
		r.GradValue.PropagateGrad(gx)
	}
}

// HasGrad returns true if there are accumulated gradients.
func (r *wrapper) HasGrad() bool {
	if r.wrapGrad {
		return r.GradValue.HasGrad()
	} else {
		return false
	}
}

// RequiresGrad returns true if the node requires gradients.
func (r *wrapper) RequiresGrad() bool {
	if r.wrapGrad {
		return r.GradValue.RequiresGrad()
	} else {
		return false
	}
}

// ZeroGrad set the gradients to zeros.
func (r *wrapper) ZeroGrad() {
	if r.wrapGrad {
		r.GradValue.ZeroGrad()
	}
}
