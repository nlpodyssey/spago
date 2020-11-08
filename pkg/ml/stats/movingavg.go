// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package stats

// MovingAvg provides a convenient way to calculate the moving average by adding value incrementally.
type MovingAvg struct {
	Mean     float64
	Variance float64
	Count    float64 // counts the added values
}

//Add adds the value to the moving average.
func (m *MovingAvg) Add(value float64) {
	m.Count++
	m.Mean += (2.0 / m.Count) * (value - m.Mean)
	m.Variance += (2.0 / m.Count) * ((value-m.Mean)*(value-m.Mean) - m.Variance)
}
