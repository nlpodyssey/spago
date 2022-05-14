// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	"sync/atomic"

	"github.com/nlpodyssey/spago/ag/fn"
	"github.com/nlpodyssey/spago/mat"
)

var (
	_ fn.Operand = &Wrapper{}
	_ Node       = &Wrapper{}
)

// Wrapper embeds any Node implementation disabling gradients handling and
// blocking gradients accumulation.
type Wrapper struct {
	Node
	// It's primarily useful for later associating a correct time-step
	// to this wrapper node, if needed for truncated backpropagation.
	createdAt uint64
}

// StopGrad creates a new Wrapper Node that stops the accumulated gradients from
// flowing through the wrapped Node.
func StopGrad(node Node) Node {
	return &Wrapper{
		Node:      node,
		createdAt: atomic.LoadUint64(&tsCounter),
	}
}

// Grad always returns nil on a Wrapper Node.
func (r *Wrapper) Grad() mat.Matrix {
	return nil
}

// AccGrad has no effects on a Wrapper Node.
func (r *Wrapper) AccGrad(mat.Matrix) {}

// HasGrad always returns false on a Wrapper Node.
func (r *Wrapper) HasGrad() bool {
	return false
}

// RequiresGrad always returns false on a Wrapper Node.
func (r *Wrapper) RequiresGrad() bool {
	return false
}

// ZeroGrad has no effects on a Wrapper Node.
func (r *Wrapper) ZeroGrad() {}
