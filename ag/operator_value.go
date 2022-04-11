// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	"sync/atomic"

	"github.com/nlpodyssey/spago/mat"
)

// Value returns the result of the function.
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

// releaseValue sets the operator's value to nil releases the memory.
func (o *Operator[T]) releaseValue() {
	o.Value() // wait for the forward goroutine to finish
	value := o.value.Load()
	if value == nil {
		return
	}
	mat.ReleaseMatrix(value.(mat.Matrix[T]))
	o.value = atomic.Value{}
}
