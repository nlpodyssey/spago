// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gbst

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestModel_Forward(t *testing.T) {
	model := newTestModel()
	g := ag.NewGraph()
	proc := nn.ReifyForTraining(model, g)

	// == Forward
	sequence := []ag.Node{
		g.NewVariable(mat.NewVecDense([]mat.Float{0.5234, 0.8113}), true),
		g.NewVariable(mat.NewVecDense([]mat.Float{-1.7743, -0.5153}), true),
		g.NewVariable(mat.NewVecDense([]mat.Float{0.9396, -1.6837}), true),
		g.NewVariable(mat.NewVecDense([]mat.Float{-0.2936, 0.3536}), true),
		g.NewVariable(mat.NewVecDense([]mat.Float{-0.5461, 0.6564}), true),
		g.NewVariable(mat.NewVecDense([]mat.Float{-0.7707, -1.346}), true),
		g.NewVariable(mat.NewVecDense([]mat.Float{1.8908, 0.2333}), true),
		g.NewVariable(mat.NewVecDense([]mat.Float{-0.1716, -0.8175}), true),
	}

	y := proc.Forward(sequence...)
	assert.InDeltaSlice(t, []mat.Float{0.40308161, -0.0114564}, y[0].Value().Data(), 1.0e-05)
	assert.InDeltaSlice(t, []mat.Float{0.56553651, 0.210189423}, y[1].Value().Data(), 1.0e-05)
	assert.InDeltaSlice(t, []mat.Float{0.23214475, -0.20046316}, y[2].Value().Data(), 1.0e-05)
	assert.InDeltaSlice(t, []mat.Float{0.29265471, -0.17325710}, y[3].Value().Data(), 1.0e-05)
	assert.InDeltaSlice(t, []mat.Float{0.61478526, 0.207365217}, y[4].Value().Data(), 1.0e-05)
	assert.InDeltaSlice(t, []mat.Float{0.57548401, 0.177836613}, y[5].Value().Data(), 1.0e-05)
	assert.InDeltaSlice(t, []mat.Float{0.22608360, -0.202010417}, y[6].Value().Data(), 1.0e-05)
	assert.InDeltaSlice(t, []mat.Float{0.33525165, -0.079017393}, y[7].Value().Data(), 1.0e-05)
}

func TestModel_ForwardScoreConsensusAttention(t *testing.T) {
	model := newTestModel()
	model.Config.ScoreConsensusAttention = true
	g := ag.NewGraph()
	proc := nn.ReifyForTraining(model, g)

	// == Forward
	sequence := []ag.Node{
		g.NewVariable(mat.NewVecDense([]mat.Float{0.5234, 0.8113}), true),
		g.NewVariable(mat.NewVecDense([]mat.Float{-1.7743, -0.5153}), true),
		g.NewVariable(mat.NewVecDense([]mat.Float{0.9396, -1.6837}), true),
		g.NewVariable(mat.NewVecDense([]mat.Float{-0.2936, 0.3536}), true),
		g.NewVariable(mat.NewVecDense([]mat.Float{-0.5461, 0.6564}), true),
		g.NewVariable(mat.NewVecDense([]mat.Float{-0.7707, -1.346}), true),
		g.NewVariable(mat.NewVecDense([]mat.Float{1.8908, 0.2333}), true),
		g.NewVariable(mat.NewVecDense([]mat.Float{-0.1716, -0.8175}), true),
	}

	y := proc.Forward(sequence...)
	assert.InDeltaSlice(t, []mat.Float{0.39854191, -0.017654827}, y[0].Value().Data(), 1.0e-05)
	assert.InDeltaSlice(t, []mat.Float{0.55043263, 0.1905072731}, y[1].Value().Data(), 1.0e-05)
	assert.InDeltaSlice(t, []mat.Float{0.22579388, -0.2080387961}, y[2].Value().Data(), 1.0e-05)
	assert.InDeltaSlice(t, []mat.Float{0.28493634, -0.1827686560}, y[3].Value().Data(), 1.0e-05)
	assert.InDeltaSlice(t, []mat.Float{0.61065015, 0.2028816025}, y[4].Value().Data(), 1.0e-05)
	assert.InDeltaSlice(t, []mat.Float{0.57294759, 0.174746066}, y[5].Value().Data(), 1.0e-05)
	assert.InDeltaSlice(t, []mat.Float{0.21875569, -0.210059952}, y[6].Value().Data(), 1.0e-05)
	assert.InDeltaSlice(t, []mat.Float{0.33201670, -0.082668955}, y[7].Value().Data(), 1.0e-05)
}

func TestModel_ForwardDownsampled(t *testing.T) {
	model := newTestModel()
	model.Config.DownsampleFactor = 4
	g := ag.NewGraph()
	proc := nn.ReifyForTraining(model, g)

	// == Forward
	sequence := []ag.Node{
		g.NewVariable(mat.NewVecDense([]mat.Float{0.5234, 0.8113}), true),
		g.NewVariable(mat.NewVecDense([]mat.Float{-1.7743, -0.5153}), true),
		g.NewVariable(mat.NewVecDense([]mat.Float{0.9396, -1.6837}), true),
		g.NewVariable(mat.NewVecDense([]mat.Float{-0.2936, 0.3536}), true),
		g.NewVariable(mat.NewVecDense([]mat.Float{-0.5461, 0.6564}), true),
		g.NewVariable(mat.NewVecDense([]mat.Float{-0.7707, -1.346}), true),
		g.NewVariable(mat.NewVecDense([]mat.Float{1.8908, 0.2333}), true),
		g.NewVariable(mat.NewVecDense([]mat.Float{-0.1716, -0.8175}), true),
	}

	y := proc.Forward(sequence...)
	assert.InDeltaSlice(t, []mat.Float{0.3733543, -0.043746825}, y[0].Value().Data(), 1.0e-05)
	assert.InDeltaSlice(t, []mat.Float{0.4379011, 0.0260435049}, y[1].Value().Data(), 1.0e-05)
}

func newTestModel() *Model {
	c := Config{
		InputSize:               2,
		MaxBlockSize:            4,
		BlockSize:               []int{1, 2, 3, 4},
		DownsampleFactor:        1,
		ScoreConsensusAttention: false,
	}
	model := New(c)
	model.Conv[0].K[0].Value().SetData([]mat.Float{-0.4808, 0.1073, 0.4607, 0.0709})
	model.Conv[1].K[0].Value().SetData([]mat.Float{0.3772, 0.3505, -0.0531, -0.2144})
	model.Conv[0].B[0].Value().SetData([]mat.Float{0.4983})
	model.Conv[1].B[0].Value().SetData([]mat.Float{0.0743})
	model.Proj[0].K[0].Value().SetData([]mat.Float{0.4217, -0.2454})
	model.Proj[1].K[0].Value().SetData([]mat.Float{0.458, -0.426})
	model.Proj[0].B[0].Value().SetData([]mat.Float{0.1281})
	model.Proj[1].B[0].Value().SetData([]mat.Float{-0.3309})
	model.Scorer.W.Value().SetData([]mat.Float{-0.1683, 0.3895})
	model.Scorer.B.Value().SetData([]mat.Float{-0.2898})
	return model
}
