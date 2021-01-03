// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hyperbolic

import mat "github.com/nlpodyssey/spago/pkg/mat32"

// Hyperbolic defines an hyperbolic decay depending on the time step
//     lr = lr / (1 + rate*t).
type Hyperbolic struct {
	init  mat.Float
	final mat.Float
	rate  mat.Float
}

// New returns a new Hyperbolic decay optimizer.
func New(init, final, rate mat.Float) *Hyperbolic {
	if init < final {
		panic("decay: the initial learning rate must be >= than the final one")
	}
	return &Hyperbolic{
		init:  init,
		final: final,
		rate:  rate,
	}
}

// Decay calculates the decay of the learning rate lr at time t.
func (d *Hyperbolic) Decay(lr mat.Float, t int) mat.Float {
	if t > 1 && d.rate > 0.0 && lr > d.final {
		return d.init / (1.0 + d.rate*mat.Float(t))
	}
	return lr
}
