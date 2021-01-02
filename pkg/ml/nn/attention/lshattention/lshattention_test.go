// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lshattention

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestModel_LshAttention(t *testing.T) {
	model := newTestModel()
	g := ag.NewGraph()
	ctx := nn.Context{Graph: g, Mode: nn.Training}
	proc := nn.Reify(ctx, model).(*Model)

	x1 := g.NewVariable(mat.NewVecDense([]mat.Float{0.3, 0.5, -0.8, 0.1, 0.3}), true)
	x2 := g.NewVariable(mat.NewVecDense([]mat.Float{-0.2, 0.4, -0.6, -0.2, 0.9}), true)
	x3 := g.NewVariable(mat.NewVecDense([]mat.Float{0.1, 0.2, 0.3, 0.0, -0.1}), true)
	x4 := g.NewVariable(mat.NewVecDense([]mat.Float{0.2, 0.4, 0.6, 0.2, 0.6}), true)
	x5 := g.NewVariable(mat.NewVecDense([]mat.Float{0.2, -0.6, -0.6, 0.9, 0.9}), true)
	x6 := g.NewVariable(mat.NewVecDense([]mat.Float{-0.5, 0.4, 0.0, 0.1, 0.8}), true)
	x7 := g.NewVariable(mat.NewVecDense([]mat.Float{0.3, 0.3, -0.8, 0.0, 0.2}), true)
	x8 := g.NewVariable(mat.NewVecDense([]mat.Float{-0.7, 0.4, 0.3, -0.2, 0.7}), true)
	x9 := g.NewVariable(mat.NewVecDense([]mat.Float{-0.8, 0.7, 0.3, -0.7, 0.0}), true)

	output := proc.Forward(x1, x2, x3, x4, x5, x6, x7, x8, x9)

	assert.InDeltaSlice(t, []mat.Float{-0.0996905, 0.7098312, 0.49985933, -1.1750140}, output[0].Value().Data(), 1.0e-05)
	assert.InDeltaSlice(t, []mat.Float{0.15378995, 0.37251139, 0.42739101, -0.46110926}, output[1].Value().Data(), 1.0e-05)
	assert.InDeltaSlice(t, []mat.Float{-0.15323142, 0.10676857, 0.01016588, -0.59778175}, output[2].Value().Data(), 1.0e-05)
	assert.InDeltaSlice(t, []mat.Float{0.1580695, 0.3598185, 0.41276363, -0.454396291}, output[3].Value().Data(), 1.0e-05)
	assert.InDeltaSlice(t, []mat.Float{-0.14, 0.84, 0.48, -0.65}, output[4].Value().Data(), 1.0e-05)
	assert.InDeltaSlice(t, []mat.Float{0.1583759, 0.35892780, 0.41173471, -0.4539261}, output[5].Value().Data(), 1.0e-05)
	assert.InDeltaSlice(t, []mat.Float{-0.1002830, 0.7101543, 0.5001286, -1.1749871}, output[6].Value().Data(), 1.0e-05)
	assert.InDeltaSlice(t, []mat.Float{0.1608603, 0.3508395, 0.4026476, -0.4492517}, output[7].Value().Data(), 1.0e-05)
	assert.InDeltaSlice(t, []mat.Float{-0.1584371, 0.10156288, 0.02439478, -0.57487669}, output[8].Value().Data(), 1.0e-05)

	output[0].PropagateGrad(mat.NewVecDense([]mat.Float{-0.04, 0.36, 0.32, 0.01}))

	g.Backward(output[0])

	assert.InDeltaSlice(t, []mat.Float{0.07993073, -0.17005416, -0.1586937, -0.2616235, 0.1777840}, x1.Grad().Data(), 1.0e-05)

	assert.InDeltaSlice(t, []mat.Float{
		-0.012, -0.0160112, 0.032, -0.0020056, -0.01000562,
		0.108, 0.14410127, -0.288, 0.018050638, 0.09005063,
		0.096, 0.12809002, -0.256, 0.016045011, 0.0800450,
		0.003, 0.0040028, -0.008, 0.0005014, 0.002501406,
	}, model.Value.W.Grad().(*mat.Dense).Data(), 1.0e-05)
	assert.InDeltaSlice(t, []mat.Float{
		-0.04, 0.36, 0.32, 0.01,
	}, model.Value.B.Grad().Data(), 1.0e-05)
	assert.InDeltaSlice(t, []mat.Float{
		1.3194545e-05, -8.789967e-05, -3.518545566e-05, -5.054710927e-05, -4.1750745e-05,
		4.3255563e-05, -0.000100, -0.000115348, -7.16580691e-05, -4.28210269e-05,
		7.6290542e-06, -0.000110, -2.03441446e-05, -5.9124849e-05, -5.40388133e-05,
		-8.932830e-06, 0.00016517, 2.382088001e-05, 8.70514586e-05, 8.10962386e-05,
	}, model.Query.W.Grad().(*mat.Dense).Data(), 1.0e-05)
	assert.InDeltaSlice(t, []mat.Float{
		4.3981819e-05, 0.000144185, 2.5430180e-05, -2.97761e-05,
	}, model.Query.B.Grad().Data(), 1.0e-05)
}

func newTestModel() *Model {
	model := New(Config{
		InputSize:   5,
		QuerySize:   4,
		BucketSize:  3,
		ValueSize:   4,
		ScaleFactor: 0.5,
	})
	model.Value.W.Value().SetData([]mat.Float{
		0.2, 0.7, 0.3, 0.5, 0.3,
		0.4, -0.5, -0.4, -0.6, 0.4,
		0.1, -0.4, -0.5, -0.9, 0.7,
		-0.9, -0.2, 0.1, 0.1, 0.2,
	})
	model.Value.B.Value().SetData([]mat.Float{-0.3, 0.4, 0.1, -0.8})
	model.Query.W.Value().SetData([]mat.Float{
		-0.3, 0.3, 0.6, -0.7, 0.7,
		0.2, 0.8, -0.1, -0.2, 0.4,
		0.4, -0.1, -0.3, 0.1, 0.5,
		0.5, 0.1, -0.8, 0.2, -0.9,
	})
	model.Query.B.Value().SetData([]mat.Float{0.3, 0.5, -0.8, 0.0})
	model.R.Value().SetData([]mat.Float{
		0.3, 0.2, 0.1,
		-0.4, -0.5, -0.3,
		-0.6, -0.4, -0.2,
		-0.2, -0.6, 0.8,
	})
	return model
}
