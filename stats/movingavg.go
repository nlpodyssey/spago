// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package stats

import "github.com/nlpodyssey/spago/mat"

// MovingAvg provides a convenient way to calculate the moving average by adding value incrementally.
type MovingAvg[T mat.DType] struct {
	Mean     T
	Variance T
	Count    T // counts the added values
}

//Add adds the value to the moving average.
func (m *MovingAvg[T]) Add(value T) {
	m.Count++
	m.Mean += (2.0 / m.Count) * (value - m.Mean)
	m.Variance += (2.0 / m.Count) * ((value-m.Mean)*(value-m.Mean) - m.Variance)
}
