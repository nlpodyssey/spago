// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat

// Value returns the value of the Matrix itself.
func (d *Dense[T]) Value() Tensor {
	return d
}

// Grad returns the gradients accumulated during the backward pass.
func (d *Dense[T]) Grad() Tensor {
	d.gradMu.RLock()
	defer d.gradMu.RUnlock()
	return d.grad
}

// AccGrad accumulates the gradients.
// It accumulates the gradients even if the requiresGrad flag is false.
func (d *Dense[T]) AccGrad(grad Tensor) {
	d.gradMu.Lock()
	defer d.gradMu.Unlock()
	if d.grad == nil {
		d.grad = grad.(Matrix).Clone().(*Dense[T])
		return
	}
	d.grad.AddInPlace(grad.(Matrix))
}

// HasGrad reports whether there are accumulated gradients.
func (d *Dense[T]) HasGrad() bool {
	d.gradMu.RLock()
	defer d.gradMu.RUnlock()
	return d.grad != nil
}

// RequiresGrad reports whether the Variable requires gradients.
func (d *Dense[T]) RequiresGrad() bool {
	return d.requiresGrad
}

// SetRequiresGrad sets the requiresGrad flag.
func (d *Dense[T]) SetRequiresGrad(v bool) {
	d.requiresGrad = v
}

// ZeroGrad zeroes the gradients, setting the value of Grad to nil.
func (d *Dense[T]) ZeroGrad() {
	d.gradMu.Lock()
	defer d.gradMu.Unlock()
	d.grad = nil
}
