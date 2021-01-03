// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package decay

import mat "github.com/nlpodyssey/spago/pkg/mat32"

// Function is implemented by any value that has the Decay method.
type Function interface {
	// Decay calculates the decay of the learning rate lr at time t.
	Decay(lr mat.Float, t int) mat.Float
}
