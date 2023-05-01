// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package decay

// Function is implemented by any value that has the Decay method.
type Function interface {
	// Decay calculates the decay of the learning rate lr at time t.
	Decay(lr float64, t int) float64
}
