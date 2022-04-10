// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	"sync/atomic"

	"github.com/nlpodyssey/spago/mat"
)

// Grad returns the gradients accumulated during the backward pass.
func (o *Operator[T]) Grad() mat.Matrix[T] {
	if !o.requiresGrad {
		return nil
	}

	if o.inBackward && atomic.LoadInt64(&o.pendingGrads) > 0 {
		o.gradMx.RLock()
		defer o.gradMx.RUnlock()
	}
	return o.grad
}

// HasGrad returns true if there are accumulated gradients.
func (o *Operator[_]) HasGrad() bool {
	return o.requiresGrad && o.Grad() != nil
}

// RequiresGrad returns true if the node requires gradients.
func (o *Operator[_]) RequiresGrad() bool {
	return o.requiresGrad
}

// ZeroGrad clears the gradients.
func (o *Operator[_]) ZeroGrad() {
	if !o.requiresGrad {
		return
	}
	o.gradMx.TryLock()
	if o.grad == nil {
		return
	}
	mat.ReleaseMatrix(o.grad) // release memory
	o.grad = nil
	o.pendingGrads = 0
	o.visited = false
	o.inBackward = false
}

// AccGrad accumulates the gradients to the node itself.
func (o *Operator[T]) AccGrad(grad mat.Matrix[T]) {
	if !o.requiresGrad {
		return
	}
	o.gradAccMx.Lock()
	defer o.gradAccMx.Unlock()

	if o.grad == nil {
		o.grad = o.Value().ZerosLike()
	}
	o.grad.AddInPlace(grad)

	if o.inBackward && atomic.AddInt64(&o.pendingGrads, -1) == 0 { // decrement
		o.gradMx.Unlock()
	}
}
