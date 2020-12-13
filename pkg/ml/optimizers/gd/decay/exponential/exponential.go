// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package exponential

import "math"

// Exponential defines an exponential decay depending on the time step:
//     lr = exp((times - t) * log(lr) + log(final))
type Exponential struct {
	init  float64
	final float64
	times int
}

func New(init, final float64, iter int) *Exponential {
	if init < final {
		panic("decay: the initial learning rate must be >= than the final one")
	}
	return &Exponential{
		init:  init,
		final: final,
		times: iter,
	}
}

func (d *Exponential) Decay(lr float64, t int) float64 {
	if t > 1 && lr > d.final {
		return math.Exp((float64(d.times-t)*math.Log(lr) + math.Log(d.final)) / float64(d.times-t+1))
	}
	return lr
}
