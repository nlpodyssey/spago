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

	if atomic.LoadInt64(&o.pendingGrads) == 0 {
		return o.grad
	}

	o.gradCond.L.Lock()
	defer o.gradCond.L.Unlock()
	for {
		if atomic.LoadInt64(&o.pendingGrads) == 0 {
			return o.grad
		}
		o.gradCond.Wait()
	}
}

// HasGrad returns true if there are accumulated gradients.
func (o *Operator[_]) HasGrad() bool {
	return o.Grad() != nil
}

// RequiresGrad returns true if the node requires gradients.
func (o *Operator[_]) RequiresGrad() bool {
	return o.requiresGrad
}

// ZeroGrad clears the gradients.
func (o *Operator[_]) ZeroGrad() {
	o.Grad() // safety wait for the backward goroutine to finish
	if o.grad == nil {
		return
	}
	mat.ReleaseMatrix(o.grad)
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
	o.gradCond.L.Lock()
	defer o.gradCond.L.Unlock()

	if o.grad == nil {
		o.grad = o.Value().ZerosLike()
	}
	o.grad.AddInPlace(grad)

	if o.inBackward && atomic.AddInt64(&o.pendingGrads, -1) == 0 {
		o.gradCond.Broadcast()
	}
}

func (o *Operator[T]) initOutputGrad(outputGrad mat.Matrix[T]) {
	if outputGrad != nil && o.grad != nil {
		panic("ag: attempt to set output gradients on a node that already has gradients")
	}

	if o.grad != nil {
		o.pendingGrads--
		return
	}

	if outputGrad != nil {
		o.AccGrad(outputGrad)
		return
	}

	gx := o.Value().OnesLike()
	o.AccGrad(gx)
	mat.ReleaseMatrix(gx)
}
