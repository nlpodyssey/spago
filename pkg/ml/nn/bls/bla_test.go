// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bls

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/stretchr/testify/assert"
	"testing"
)

func Test_ridgeRegression(t *testing.T) {
	t.Run("float32", testridgeRegression[float32])
	t.Run("float64", testridgeRegression[float64])
}

func testridgeRegression[T mat.DType](t *testing.T) {
	x := mat.NewDense(5, 4, []T{
		0.3, 0.3, 0.0, 0.3,
		0.7, -0.5, -0.1, -0.9,
		0.2, 0.8, 0.3, 0.6,
		0.5, 0.6, 0.7, 0.4,
		0.8, 0.0, -0.2, 0.7,
	})
	y := mat.NewDense(5, 2, []T{
		0.7, -0.3,
		0.7, 0.4,
		0.6, -0.1,
		-0.9, 0.5,
		-0.2, 0.5,
	})

	b := ridgeRegression[T](x, y, 0.9)

	assert.InDeltaSlice(t, []T{
		0.13276582, 0.3392879,
		0.13017280, -0.101273,
		-0.3292762, 0.1246170,
		-0.2256778, 0.0004106,
	}, b.Data(), 1.0e-5)
}

func Test_admn(t *testing.T) {
	t.Run("float32", testadmn[float32])
	t.Run("float64", testadmn[float64])
}

func testadmn[T mat.DType](t *testing.T) {
	a := mat.NewDense(5, 3, []T{
		0.3, 0.3, 0.0,
		0.7, -0.5, -0.1,
		0.2, 0.8, 0.3,
		0.5, 0.6, 0.7,
		0.8, 0.0, -0.2,
	})
	xaugm := mat.NewDense(5, 2, []T{
		0.7, -0.3,
		0.7, 0.4,
		0.6, -0.1,
		-0.9, 0.5,
		-0.2, 0.5,
	})

	b := admn[T](a, xaugm, 0.001, 2)

	assert.InDeltaSlice(t, []T{
		0.15778322, 0.4546243,
		0.05815867, -0.169100,
		-0.5242062, 0.2030716,
	}, b.Data(), 1.0e-5)
}
