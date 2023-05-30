// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package radam

import (
	"testing"

	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/stretchr/testify/assert"
)

func TestRAdam_DeltaTimeStep1(t *testing.T) {
	t.Run("float32", testRAdamDeltaTimeStep1[float32])
	t.Run("float64", testRAdamDeltaTimeStep1[float64])
}

func testRAdamDeltaTimeStep1[T float.DType](t *testing.T) {
	updater := New[T](NewConfig(
		0.001,  // step size
		0.9,    // beta1
		0.999,  // beta2
		1.0e-8, // epsilon
	))

	params := mat.NewDense[T](mat.WithBacking([]T{0.4, 0.4, 0.5, 1.0, 0.8}))
	grads := mat.NewDense[T](mat.WithBacking([]T{0.9, 0.7, 0.4, 0.8, 0.1}))

	supp := updater.newState(params.Shape()...)
	mat.SetData[T](supp.M, []T{0.7, 0.8, 0.5, 0.3, 0.2})
	mat.SetData[T](supp.V, []T{1.0, 0.4, 0.7, 0.0, 0.2})

	params.SubInPlace(updater.calculateParamUpdate(grads, supp))

	assert.InDeltaSlice(t, []T{0.399772, 0.399605, 0.499815, 0.995625, 0.799866}, params.Data(), 1.0e-6)
}

func TestRAdam_DeltaTimeStep6(t *testing.T) {
	t.Run("float32", testRAdaDeltaTimeStep6[float32])
	t.Run("float64", testRAdaDeltaTimeStep6[float64])
}

func testRAdaDeltaTimeStep6[T float.DType](t *testing.T) {
	updater := New[T](NewConfig(
		0.001,  // step size
		0.9,    // beta1
		0.999,  // beta2
		1.0e-8, // epsilon
	))

	params := mat.NewDense[T](mat.WithBacking([]T{0.4, 0.4, 0.5, 1.0, 0.8}))
	grads := mat.NewDense[T](mat.WithBacking([]T{0.9, 0.7, 0.4, 0.8, 0.1}))

	supp := updater.newState(params.Shape()...)
	mat.SetData[T](supp.M, []T{0.7, 0.8, 0.5, 0.3, 0.2})
	mat.SetData[T](supp.V, []T{1.0, 0.4, 0.7, 0.0, 0.2})

	for i := 0; i < 5; i++ {
		updater.IncBatch()
	}

	if updater.TimeStep != 6 {
		t.Error("The time-step doesn't match the expected value")
	}

	params.SubInPlace(updater.calculateParamUpdate(grads, supp))

	assert.InDeltaSlice(t, []T{0.399997, 0.399995, 0.499998, 0.999941, 0.799998}, params.Data(), 1.0e-6)
}
