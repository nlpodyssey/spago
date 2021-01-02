// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package adagrad

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/stretchr/testify/assert"
	"testing"
)

func Test_Update(t *testing.T) {
	updater := New(NewConfig(0.001, 1.0e-8))
	params := mat.NewVecDense([]mat.Float{0.4, 0.4, 0.5, 1.0, 0.8})
	grads := mat.NewVecDense([]mat.Float{0.9, 0.7, 0.4, 0.8, 0.1})
	supp := updater.NewSupport(params.Dims()).Data
	supp[m].SetData([]mat.Float{1.0, 0.4, 0.7, 0.0, 0.2})

	params.SubInPlace(updater.calcDelta(grads, supp))

	assert.InDeltaSlice(t, []mat.Float{0.399331, 0.399258, 0.499569, 0.999, 0.799782}, params.Data(), 1.0e-6)
}

func Test_Update2(t *testing.T) {
	updater := New(NewConfig(
		0.001,  // step size
		1.0e-8, // epsilon
	))

	params := mat.NewDense(3, 3, []mat.Float{
		1.4, 1.3, 0,
		-0.8, 0.16, 0.65,
		0.7, -0.4, 0.2,
	})

	grads := mat.NewDense(3, 3, []mat.Float{
		0.5, 0.3, -0.1,
		-0.6, -0.4, -1.0,
		0.5, -0.6, 0.1,
	})

	supp := updater.NewSupport(params.Dims()).Data

	// === First iteration

	params.SubInPlace(updater.calcDelta(grads, supp))

	assert.InDeltaSlice(t, []mat.Float{
		0.25, 0.09, 0.01,
		0.36, 0.16, 1,
		0.25, 0.36, 0.01,
	}, supp[m].Data(), 1.0e-6)

	assert.InDeltaSlice(t, []mat.Float{
		1.39900000002, 1.29900000003333, 0.0009999999,
		-0.799000000016667, 0.160999999975, 0.65099999999,
		0.69900000002, -0.399000000016667, 0.1990000001,
	}, params.Data(), 1.0e-6)

	// === Second iteration

	grads2 := mat.NewDense(3, 3, []mat.Float{
		0.7, 0.44, -0.66,
		-0.56, 0.4, 1.4,
		0.44, 1.44, 2.44,
	})

	params.SubInPlace(updater.calcDelta(grads2, supp))

	assert.InDeltaSlice(t, []mat.Float{
		0.74, 0.2836, 0.4456,
		0.6736, 0.32, 2.96,
		0.4436, 2.4336, 5.9636,
	}, supp[m].Data(), 1.0e-6)

	assert.InDeltaSlice(t, []mat.Float{
		1.39818626655825, 1.29817377270604, 0.001988715389413,
		-0.798317681774621, 0.160292893206313, 0.650186266523523,
		0.698339372135027, -0.399923076933827, 0.198000838875607,
	}, params.Data(), 1.0e-5)
}
