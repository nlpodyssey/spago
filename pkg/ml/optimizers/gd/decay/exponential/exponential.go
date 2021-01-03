// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package exponential

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
)

// Exponential defines an exponential decay depending on the time step:
//     lr = exp((times - t) * log(lr) + log(final))
type Exponential struct {
	init  mat.Float
	final mat.Float
	times int
}

// New returns a new Exponential decay optimizer.
func New(init, final mat.Float, iter int) *Exponential {
	if init < final {
		panic("decay: the initial learning rate must be >= than the final one")
	}
	return &Exponential{
		init:  init,
		final: final,
		times: iter,
	}
}

// Decay calculates the decay of the learning rate lr at time t.
func (d *Exponential) Decay(lr mat.Float, t int) mat.Float {
	if t > 1 && lr > d.final {
		return mat.Exp((mat.Float(d.times-t)*mat.Log(lr) + mat.Log(mat.Float(d.final))) / mat.Float(d.times-t+1))
	}
	return lr
}
