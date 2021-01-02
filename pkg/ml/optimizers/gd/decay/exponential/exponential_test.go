// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package exponential

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestNewExponential_Decay(t *testing.T) {
	fn := New(0.01, 0.001, 10)

	assert.InDelta(t, 0.01, fn.Decay(0.01, 1), 1.0e-06)
	assert.InDelta(t, 0.00774264, fn.Decay(0.01, 2), 1.0e-06)
	assert.InDelta(t, 0.00599484, fn.Decay(0.00774263682, 3), 1.0e-06)
	assert.InDelta(t, 0.001, fn.Decay(0.001, 10), 1.0e-06)
}

func TestNew(t *testing.T) {
	assert.NotPanics(t, func() { New(0.01, 0.001, 10) }, "The New did panic unexpectedly")
	assert.Panics(t, func() { New(0.001, 0.01, 10) }, "The New had to panic with init lr < final lr")
}
