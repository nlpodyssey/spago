// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package selfattention

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/attention"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestModel_SelfAttention(t *testing.T) {
	model := newTestModel()
	g := ag.NewGraph()
	proc := nn.Reify(nn.Context{Graph: g, Mode: nn.Training}, model).(*Model)

	x1 := g.NewVariable(mat.NewVecDense([]mat.Float{-0.8, -0.9, -0.9, 1.0}), true)
	x2 := g.NewVariable(mat.NewVecDense([]mat.Float{0.8, -0.3, 0.5, 0.3}), true)
	x3 := g.NewVariable(mat.NewVecDense([]mat.Float{-0.2, 0.7, 0.2, 0.4}), true)

	output := proc.Forward(attention.ToQKV([]ag.Node{x1, x2, x3}))

	assert.InDeltaSlice(t, []mat.Float{0.789110, -0.755551, -0.431247}, output[0].Value().Data(), 1.0e-05)
	assert.InDeltaSlice(t, []mat.Float{0.780654, -0.6212001, -0.380214}, output[1].Value().Data(), 1.0e-05)
	assert.InDeltaSlice(t, []mat.Float{0.7586521, -0.569575, -0.390976}, output[2].Value().Data(), 1.0e-05)

	output[0].PropagateGrad(mat.NewVecDense([]mat.Float{-0.04, 0.36, 0.32}))
	output[1].PropagateGrad(mat.NewVecDense([]mat.Float{-0.08, -0.2, -0.1}))
	output[2].PropagateGrad(mat.NewVecDense([]mat.Float{0.1, 0.3, 0.8}))

	g.BackwardAll()

	assert.InDeltaSlice(t, []mat.Float{0.01654154, 0.48942297, -0.1587743, -0.2387454}, x1.Grad().Data(), 1.0e-05)
	assert.InDeltaSlice(t, []mat.Float{0.04408108, 0.18716132, -0.15425818, -0.040870}, x2.Grad().Data(), 1.0e-05)
	assert.InDeltaSlice(t, []mat.Float{0.05800791, 0.205784865, -0.2431444, -0.1281430}, x3.Grad().Data(), 1.0e-05)

	assert.InDeltaSlice(t, []mat.Float{
		0.01790323, 0.01853984, 0.01959262, -0.020789988,
		-0.1529254, -0.2133677, -0.17563661, 0.336455424,
		-0.2887047, -0.3777314, -0.31337569, 0.705232107,
	}, model.Value.W.Grad().(*mat.Dense).Data(), 1.0e-05)
	assert.InDeltaSlice(t, []mat.Float{
		-0.02, 0.46, 1.02,
	}, model.Value.B.Grad().Data(), 1.0e-05)
	assert.InDeltaSlice(t, []mat.Float{
		0.07438275, 0.15194683, 0.11696175, -0.0629919,
		0.03235329, 0.05018469, 0.04422187, -0.0234946,
		-0.0599427, -0.1594204, -0.1097165, 0.0598379,
	}, model.Key.W.Grad().(*mat.Dense).Data(), 1.0e-05)
	assert.InDeltaSlice(t, []mat.Float{
		0, 0, 0,
	}, model.Key.B.Grad().Data(), 1.0e-05)
	assert.InDeltaSlice(t, []mat.Float{
		0.00538138, -0.0264289, -0.0085512, -0.0088408,
		-0.0175901, -0.0032803, -0.0132455, 0.0143783,
		-0.1022365, -0.0221910, -0.0784209, 0.08306303,
	}, model.Query.W.Grad().(*mat.Dense).Data(), 1.0e-05)
	assert.InDeltaSlice(t, []mat.Float{
		-0.0267918, 0.0149118, 0.08413719,
	}, model.Query.B.Grad().Data(), 1.0e-05)
}

func newTestModel() *Model {
	model := New(Config{
		InputSize:   4,
		QuerySize:   3,
		KeySize:     3,
		ValueSize:   3,
		ScaleFactor: 1.0 / mat.Sqrt(3.0),
	})
	model.Value.W.Value().SetData([]mat.Float{
		0.5, 0.6, -0.8, 0.7,
		-0.4, 0.1, 0.7, -0.7,
		0.3, 0.8, -0.9, 0.0,
	})
	model.Value.B.Value().SetData([]mat.Float{0.4, 0.0, -0.3})
	model.Key.W.Value().SetData([]mat.Float{
		0.7, -0.2, -0.1, 0.2,
		-0.1, -0.1, 0.3, -0.2,
		0.6, 0.1, 0.9, 0.3,
	})
	model.Key.B.Value().SetData([]mat.Float{0.8, -0.2, -0.5})
	model.Query.W.Value().SetData([]mat.Float{
		-0.8, -0.6, 0.2, 0.5,
		0.7, -0.6, -0.3, 0.6,
		-0.3, 0.3, 0.4, -0.8,
	})
	model.Query.B.Value().SetData([]mat.Float{0.3, 0.5, -0.7})
	return model
}
