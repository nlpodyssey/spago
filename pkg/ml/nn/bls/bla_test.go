// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bls

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/stretchr/testify/assert"
	"testing"
)

func Test_ridgeRegression(t *testing.T) {
	x := mat.NewDense(5, 4, []mat.Float{
		0.3, 0.3, 0.0, 0.3,
		0.7, -0.5, -0.1, -0.9,
		0.2, 0.8, 0.3, 0.6,
		0.5, 0.6, 0.7, 0.4,
		0.8, 0.0, -0.2, 0.7,
	})
	y := mat.NewDense(5, 2, []mat.Float{
		0.7, -0.3,
		0.7, 0.4,
		0.6, -0.1,
		-0.9, 0.5,
		-0.2, 0.5,
	})

	b := ridgeRegression(x, y, 0.9)

	assert.InDeltaSlice(t, []mat.Float{
		0.13276582, 0.3392879,
		0.13017280, -0.101273,
		-0.3292762, 0.1246170,
		-0.2256778, 0.0004106,
	}, b.Data(), 1.0e-5)
}

func Test_admn(t *testing.T) {
	a := mat.NewDense(5, 3, []mat.Float{
		0.3, 0.3, 0.0,
		0.7, -0.5, -0.1,
		0.2, 0.8, 0.3,
		0.5, 0.6, 0.7,
		0.8, 0.0, -0.2,
	})
	xaugm := mat.NewDense(5, 2, []mat.Float{
		0.7, -0.3,
		0.7, 0.4,
		0.6, -0.1,
		-0.9, 0.5,
		-0.2, 0.5,
	})

	b := admn(a, xaugm, 0.001, 2)

	assert.InDeltaSlice(t, []mat.Float{
		0.15778322, 0.4546243,
		0.05815867, -0.169100,
		-0.5242062, 0.2030716,
	}, b.Data(), 1.0e-5)
}
