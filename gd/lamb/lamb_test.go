// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lamb

import (
	"testing"

	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/stretchr/testify/assert"
)

func Test_IncExample(t *testing.T) {
	t.Run("float32", testIncExample[float32])
	t.Run("float64", testIncExample[float64])
}

func testIncExample[T float.DType](t *testing.T) {
	updater := New[T](NewConfig(
		0.001,  // step size
		0.9,    // beta1
		0.999,  // beta2
		1.0e-8, // epsilon
		0.1,    // lambda
	))
	assert.InDelta(t, 3.1623e-4, updater.Alpha, 1.0e-08)
}

func Test_Update(t *testing.T) {
	t.Run("float32", testUpdate[float32])
	t.Run("float64", testUpdate[float64])
}

func testUpdate[T float.DType](t *testing.T) {
	updater := New[T](NewConfig(
		0.001,  // step size
		0.9,    // beta1
		0.999,  // beta2
		1.0e-8, // epsilon
		0.1,    // lambda
	))

	params := mat.NewVecDense([]T{0.4, 0.4, 0.5, 1.0, 0.8})
	grads := mat.NewVecDense([]T{0.9, 0.7, 0.4, 0.8, 0.1})

	supp := updater.NewSupport(params.Dims()).Data
	mat.SetData[T](supp[v], []T{0.7, 0.8, 0.5, 0.3, 0.2})
	mat.SetData[T](supp[m], []T{1.0, 0.4, 0.7, 0.0, 0.2})

	params.SubInPlace(updater.calcDelta(grads, supp, params))

	assert.InDeltaSlice(t, []T{0.399975, 0.399957, 0.499979, 0.999533, 0.799983}, params.Data(), 1.0e-6)
}

func Test_Update2(t *testing.T) {
	t.Run("float32", testUpdate2[float32])
	t.Run("float64", testUpdate2[float64])
}

func testUpdate2[T float.DType](t *testing.T) {
	updater := New[T](NewConfig(
		0.001,  // step size
		0.9,    // beta1
		0.999,  // beta2
		1.0e-8, // epsilon
		0.1,    // lambda
	))

	params := mat.NewDense(3, 3, []T{
		1.4, 1.3, 0,
		-0.8, 0.16, 0.65,
		0.7, -0.4, 0.2,
	})

	grads := mat.NewDense(3, 3, []T{
		0.5, 0.3, -0.1,
		-0.6, -0.4, -1.0,
		0.5, -0.6, 0.1,
	})

	supp := updater.NewSupport(params.Dims()).Data

	// === First iteration

	params.SubInPlace(updater.calcDelta(grads, supp, params))

	assert.InDeltaSlice(t, []T{
		0.05, 0.03, -0.01,
		-0.06, -0.04, -0.1,
		0.05, -0.06, 0.01,
	}, supp[v].Data(), 1.0e-6)

	assert.InDeltaSlice(t, []T{
		0.00025, 9.0e-05, 1e-05,
		0.00036, 0.00016, 0.001,
		0.00025, 0.00036, 1e-05,
	}, supp[m].Data(), 1.0e-6)

	assert.InDeltaSlice(t, []T{
		1.3997471, 1.2997478, 0.00024214012,
		-0.79975176, 0.16024092, 0.65023714,
		0.6997525, -0.3997548, 0.19975632,
	}, params.Data(), 1.0e-6)

	// === Second iteration

	updater.IncExample()

	grads2 := mat.NewDense(3, 3, []T{
		0.7, 0.44, -0.66,
		-0.56, 0.4, 1.4,
		0.44, 1.44, 2.44,
	})

	params.SubInPlace(updater.calcDelta(grads2, supp, params))

	assert.InDeltaSlice(t, []T{
		0.115, 0.071, -0.075,
		-0.11, 0.004, 0.05,
		0.089, 0.09, 0.253,
	}, supp[v].Data(), 1.0e-6)

	assert.InDeltaSlice(t, []T{
		0.00073975, 0.00028351, 0.00044559,
		0.00067324, 0.00031984, 0.002959,
		0.00044335, 0.00243324, 0.00596359,
	}, supp[m].Data(), 1.0e-6)

	assert.InDeltaSlice(t, []T{
		1.399511, 1.2995129, 0.00043422784,
		-0.7995182, 0.16022795, 0.6501839,
		0.69952023, -0.39985126, 0.19957812,
	}, params.Data(), 1.0e-5)
}
