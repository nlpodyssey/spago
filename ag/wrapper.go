// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	"github.com/nlpodyssey/spago/ag/fn"
	"github.com/nlpodyssey/spago/mat"
)

var (
	_ fn.Operand[float32] = &Wrapper[float32]{}
	_ GradValue[float32]  = &Wrapper[float32]{}
	_ Node[float32]       = &Wrapper[float32]{}
)

// Wrapper is a type of node.
type Wrapper[T mat.DType] struct {
	Node[T]
}

// StopGrad stops the accumulated gradient from flowing through that node in the backward direction.
func StopGrad[T mat.DType](node Node[T]) Node[T] {
	return &Wrapper[T]{
		Node: node,
	}
}

// Grad returns the gradients accumulated during the backward pass.
func (r *Wrapper[T]) Grad() mat.Matrix[T] {
	return nil
}

// AccGrad accumulates the gradients to the node.
func (r *Wrapper[T]) AccGrad(_ mat.Matrix[T]) {
	return
}

// HasGrad returns true if there are accumulated gradients.
func (r *Wrapper[_]) HasGrad() bool {
	return false
}

// RequiresGrad returns true if the node requires gradients.
func (r *Wrapper[_]) RequiresGrad() bool {
	return false
}

// ZeroGrad set the gradients to zeros.
func (r *Wrapper[_]) ZeroGrad() {
	return
}
