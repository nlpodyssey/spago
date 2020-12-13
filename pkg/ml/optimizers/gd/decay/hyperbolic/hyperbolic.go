// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hyperbolic

// Hyperbolic defines an hyperbolic decay depending on the time step
//     lr = lr / (1 + rate*t).
type Hyperbolic struct {
	init  float64
	final float64
	rate  float64
}

func New(init, final, rate float64) *Hyperbolic {
	if init < final {
		panic("decay: the initial learning rate must be >= than the final one")
	}
	return &Hyperbolic{
		init:  init,
		final: final,
		rate:  rate,
	}
}

func (d *Hyperbolic) Decay(lr float64, t int) float64 {
	if t > 1 && d.rate > 0.0 && lr > d.final {
		return d.init / (1.0 + d.rate*float64(t))
	}
	return lr
}
