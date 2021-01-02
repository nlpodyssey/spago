// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rand

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"golang.org/x/exp/rand"
)

// Float returns, as a mat.Float, a pseudo-random number in [0.0,1.0)
// from the default Source.
func Float() mat.Float {
	return rand.Float64()
}
