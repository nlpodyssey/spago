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
	_ fn.Operand[float32] = &Wrapper[float32]{}
	_ Node[float32]       = &Wrapper[float32]{}
)

// Wrapper embeds any Node implementation disabling gradients handling and
// blocking gradients accumulation.
type Wrapper[T mat.DType] struct {
	Node[T]
	// It's primarily useful for later associating a correct time-step
	// to this wrapper node, if needed for truncated backpropagation.
	createdAt uint64
}

// StopGrad creates a new Wrapper Node that stops the accumulated gradients from
// flowing through the wrapped Node.
func StopGrad[T mat.DType](node Node[T]) Node[T] {
	return &Wrapper[T]{
		Node:      node,
		createdAt: atomic.LoadUint64(&tsCounter),
	}
}

// Grad always returns nil on a Wrapper Node.
func (r *Wrapper[T]) Grad() mat.Matrix[T] {
	return nil
}

// AccGrad has no effects on a Wrapper Node.
func (r *Wrapper[T]) AccGrad(mat.Matrix[T]) {}

// HasGrad always returns false on a Wrapper Node.
func (r *Wrapper[_]) HasGrad() bool {
	return false
}

// RequiresGrad always returns false on a Wrapper Node.
func (r *Wrapper[_]) RequiresGrad() bool {
	return false
}

// ZeroGrad has no effects on a Wrapper Node.
func (r *Wrapper[_]) ZeroGrad() {}
