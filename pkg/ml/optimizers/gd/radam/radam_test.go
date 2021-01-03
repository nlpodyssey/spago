// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package radam

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestRAdam_DeltaTimeStep1(t *testing.T) {
	updater := New(NewConfig(
		0.001,  // step size
		0.9,    // beta1
		0.999,  // beta2
		1.0e-8, // epsilon
	))

	params := mat.NewVecDense([]mat.Float{0.4, 0.4, 0.5, 1.0, 0.8})
	grads := mat.NewVecDense([]mat.Float{0.9, 0.7, 0.4, 0.8, 0.1})

	supp := updater.NewSupport(params.Dims()).Data
	supp[m].SetData([]mat.Float{0.7, 0.8, 0.5, 0.3, 0.2})
	supp[v].SetData([]mat.Float{1.0, 0.4, 0.7, 0.0, 0.2})

	params.SubInPlace(updater.calcDelta(grads, supp))

	assert.InDeltaSlice(t, []mat.Float{0.399772, 0.399605, 0.499815, 0.995625, 0.799866}, params.Data(), 1.0e-6)
}

func TestRAdam_DeltaTimeStep6(t *testing.T) {
	updater := New(NewConfig(
		0.001,  // step size
		0.9,    // beta1
		0.999,  // beta2
		1.0e-8, // epsilon
	))

	params := mat.NewVecDense([]mat.Float{0.4, 0.4, 0.5, 1.0, 0.8})
	grads := mat.NewVecDense([]mat.Float{0.9, 0.7, 0.4, 0.8, 0.1})

	supp := updater.NewSupport(params.Dims()).Data
	supp[m].SetData([]mat.Float{0.7, 0.8, 0.5, 0.3, 0.2})
	supp[v].SetData([]mat.Float{1.0, 0.4, 0.7, 0.0, 0.2})

	for i := 0; i < 5; i++ {
		updater.IncBatch()
	}

	if updater.TimeStep != 6 {
		t.Error("The time-step doesn't match the expected value")
	}

	params.SubInPlace(updater.calcDelta(grads, supp))

	assert.InDeltaSlice(t, []mat.Float{0.399997, 0.399995, 0.499998, 0.999941, 0.799998}, params.Data(), 1.0e-6)
}
