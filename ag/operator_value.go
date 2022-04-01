// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	"sync/atomic"

	"github.com/nlpodyssey/spago/mat"
)

// Value returns the result of the function.
//
// If the value is null and a goroutine is not processing it (as it happens after a g.Clear(true)),
// it automatically initiates a forward operation from the last processed node to this.
func (o *Operator[T]) Value() mat.Matrix[T] {
	if v := o.value.Load(); v != nil {
		return v.(mat.Matrix[T])
	}

	o.valueCond.L.Lock()
	defer o.valueCond.L.Unlock()
	for {
		if v := o.value.Load(); v != nil {
			return v.(mat.Matrix[T])
		}
		o.valueCond.Wait()
	}
}

// ScalarValue returns the scalar value of the node.
// It panics if the value is not a scalar.
// Note that it is not possible to start the backward step from a scalar value.
func (o *Operator[T]) ScalarValue() T {
	return o.Value().Scalar()
}

// releaseValue sets the operator's value to nil releases the memory.
func (o *Operator[T]) releaseValue() {
	value := o.value.Load()
	if value == nil {
		return
	}
	mat.ReleaseMatrix(value.(mat.Matrix[T]))
	o.value = atomic.Value{}
}
