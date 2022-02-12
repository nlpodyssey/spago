// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package exponential

import (
	"github.com/nlpodyssey/spago/mat"
)

// Exponential defines an exponential decay depending on the time step:
//     lr = exp((times - t) * log(lr) + log(final))
type Exponential[T mat.DType] struct {
	init  T
	final T
	times int
}

// New returns a new Exponential decay optimizer.
func New[T mat.DType](init, final T, iter int) *Exponential[T] {
	if init < final {
		panic("decay: the initial learning rate must be >= than the final one")
	}
	return &Exponential[T]{
		init:  init,
		final: final,
		times: iter,
	}
}

// Decay calculates the decay of the learning rate lr at time t.
func (d *Exponential[T]) Decay(lr T, t int) T {
	if t > 1 && lr > d.final {
		return mat.Exp((T(d.times-t)*mat.Log(lr) + mat.Log(d.final)) / T(d.times-t+1))
	}
	return lr
}
