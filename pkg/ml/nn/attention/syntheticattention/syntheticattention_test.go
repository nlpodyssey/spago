// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntheticattention

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestModel_SyntheticAttention(t *testing.T) {
	model := newTestModel()
	g := ag.NewGraph()
	ctx := nn.Context{Graph: g, Mode: nn.Training}
	proc := nn.Reify(ctx, model).(*Model)

	x1 := g.NewVariable(mat.NewVecDense([]mat.Float{-0.8, -0.9, -0.9, 1.0}), true)
	x2 := g.NewVariable(mat.NewVecDense([]mat.Float{0.8, -0.3, 0.5, 0.3}), true)
	x3 := g.NewVariable(mat.NewVecDense([]mat.Float{-0.2, 0.7, 0.2, 0.4}), true)

	// == Forward
	output := proc.Forward(x1, x2, x3)

	assert.InDeltaSlice(t, []mat.Float{0.778722, -0.5870107, -0.36687185}, output[0].Value().Data(), 1.0e-05)
	assert.InDeltaSlice(t, []mat.Float{0.8265906, -0.7254198, -0.3560310}, output[1].Value().Data(), 1.0e-05)
	assert.InDeltaSlice(t, []mat.Float{0.7651726, -0.5680948, -0.3797024}, output[2].Value().Data(), 1.0e-05)

	// == Backward
	output[0].PropagateGrad(mat.NewVecDense([]mat.Float{-0.04, 0.36, 0.32}))
	output[1].PropagateGrad(mat.NewVecDense([]mat.Float{-0.08, -0.2, -0.1}))
	output[2].PropagateGrad(mat.NewVecDense([]mat.Float{0.1, 0.3, 0.8}))

	g.BackwardAll()

	assert.InDeltaSlice(t, []mat.Float{-0.0015012, 0.40111136, -0.26565675, -0.1677756}, x1.Grad().Data(), 1.0e-05)
	assert.InDeltaSlice(t, []mat.Float{0.0472872, 0.19752994, -0.14673837, -0.0747287}, x2.Grad().Data(), 1.0e-05)
	assert.InDeltaSlice(t, []mat.Float{0.02086887, 0.2511595, -0.1880584, -0.07230184}, x3.Grad().Data(), 1.0e-05)
	assert.InDeltaSlice(t, []mat.Float{
		0.02565823, 0.01689616, 0.02447476, -0.02306554,
		-0.0871563, -0.1083235, -0.0844748, 0.287951612,
		-0.2523685, -0.2838154, -0.2480056, 0.669627458,
	}, model.Value.W.Grad().(*mat.Dense).Data(), 1.0e-05)
	assert.InDeltaSlice(t, []mat.Float{
		-0.02, 0.46, 1.02,
	}, model.Value.B.Grad().Data(), 1.0e-05)
	assert.InDeltaSlice(t, []mat.Float{
		0.10050248, 0.03293678, 0.08198445, -0.0819892,
		0.0, 0.0, 0.0, 0.0,
		-0.0075098, 0.0028162, -0.0046936, -0.00281620,
		0.00989979, -0.034649, -0.0098997, -0.01979958,
	}, model.FFN.Layers[0].(*linear.Model).W.Grad().Data(), 1.0e-05)
	assert.InDeltaSlice(t, []mat.Float{
		-0.0779462, 0.0, -0.0093873, -0.04949895,
	}, model.FFN.Layers[0].(*linear.Model).B.Grad().Data(), 1.0e-05)
	assert.InDeltaSlice(t, []mat.Float{
		-0.0589030, 0.0, 0.02524230, -0.02234341,
		-0.0300720, 0.0, -0.0014955, -0.01170433,
		0.08897514, 0.0, -0.0237467, 0.034047752,
		0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0,
	}, model.W.Grad().Data(), 1.0e-05)
}

func newTestModel() *Model {
	model := New(Config{
		InputSize:  4,
		HiddenSize: 4,
		MaxLength:  5,
		ValueSize:  3,
	})
	model.Value.W.Value().SetData([]mat.Float{
		0.5, 0.6, -0.8, 0.7,
		-0.4, 0.1, 0.7, -0.7,
		0.3, 0.8, -0.9, 0.0,
	})
	model.Value.B.Value().SetData([]mat.Float{0.4, 0.0, -0.3})
	model.FFN.Layers[0].(*linear.Model).W.Value().SetData([]mat.Float{
		0.7, -0.2, -0.1, 0.2,
		-0.1, -0.1, 0.3, -0.2,
		0.6, 0.1, 0.9, 0.3,
		-0.3, 0.3, 0.4, -0.8,
	})
	model.FFN.Layers[0].(*linear.Model).B.Value().SetData([]mat.Float{0.8, -0.2, -0.5, 0.2})
	model.W.Value().SetData([]mat.Float{
		0.4, 0.3, 0.2, -0.5,
		-0.9, -0.4, 0.1, -0.4,
		-0.3, 0.3, 0.4, -0.8,
		0.7, 0.8, 0.2, 0.3,
		0.5, 0.2, 0.4, -0.8,
	})
	return model
}
