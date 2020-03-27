// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hyperbolic

import (
	"github.com/nlpodyssey/spago/pkg/mat/f64utils"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestHyperbolic_Decay(t *testing.T) {
	fn := New(0.01, 0.001, 0.5)

	if !f64utils.EqualApprox(fn.Decay(0.01, 1), 0.01) {
		t.Error("the new learning rate doesn't match the expected value at time step 1")
	}

	if !f64utils.EqualApprox(fn.Decay(0.01, 2), 0.005) {
		t.Error("the new learning rate doesn't match the expected value at time step 2")
	}

	if !f64utils.EqualApprox(fn.Decay(0.00774263682, 3), 0.004) {
		t.Error("the new learning rate doesn't match the expected value at time step 3")
	}

	if !f64utils.EqualApprox(fn.Decay(0.001, 10), 0.001) {
		t.Error("the new learning rate doesn't match the expected value with t>1 and init = final")
	}
}

func TestNew(t *testing.T) {
	assert.NotPanics(t, func() { New(0.01, 0.001, 0.5) }, "The New did panic unexpectedly")
	assert.Panics(t, func() { New(0.001, 0.01, 0.5) }, "The New had to panic with init lr < final lr")
}
