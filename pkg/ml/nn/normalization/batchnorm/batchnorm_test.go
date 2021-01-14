// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package batchnorm

import (
	"github.com/stretchr/testify/assert"
	"io/ioutil"
	"math/rand"
	"os"
	"testing"

	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/utils"
	"github.com/stretchr/testify/require"
)

func TestModel_Forward_Params(t *testing.T) {
	const numDataInstances = 10000
	const dataSize = 100

	testCases := []struct {
		shift          mat.Float
		multiplier     mat.Float
		momentum       mat.Float
		expectedAvg    mat.Float
		expectedStdDev mat.Float
		forwardSteps   int
	}{
		{
			multiplier:     1.0,
			momentum:       0.0,
			expectedAvg:    0.0,
			expectedStdDev: 1.0,
			forwardSteps:   1,
		},
		{
			shift:          10.0,
			multiplier:     1.0,
			momentum:       0.0,
			expectedAvg:    10.0,
			expectedStdDev: 1.0,
			forwardSteps:   1,
		},
		{
			multiplier:     1.0,
			momentum:       0.5,
			expectedAvg:    0.0,
			expectedStdDev: 0.5,
			forwardSteps:   1,
		},
		{
			multiplier:     1.0,
			momentum:       1.0,
			expectedAvg:    0.0,
			expectedStdDev: 0.0,
			forwardSteps:   1,
		},
		{
			multiplier:     2.0,
			momentum:       0.0,
			expectedAvg:    0.0,
			expectedStdDev: 2.0,
			forwardSteps:   1,
		},
		{
			multiplier:     1.0,
			momentum:       0.5,
			expectedAvg:    0.0,
			expectedStdDev: 0.5,
			forwardSteps:   1,
		},
		{
			multiplier:     1.0,
			momentum:       0.5,
			expectedAvg:    0.0,
			expectedStdDev: 0.75,
			forwardSteps:   2,
		},
	}

	rnd := rand.New(rand.NewSource(42))

	testData := make([][]mat.Float, numDataInstances)
	for i := range testData {
		testData[i] = make([]mat.Float, dataSize)
		for j := range testData[i] {
			testData[i][j] = mat.Float(rnd.NormFloat64())
		}
	}

	for _, tt := range testCases {
		model := NewWithMomentum(dataSize, tt.momentum)
		data := make([][]mat.Float, len(testData))
		for i := range testData {
			data[i] = make([]mat.Float, dataSize)
			for j := range testData[i] {
				data[i][j] = tt.multiplier*testData[i][j] + tt.shift
			}
		}

		x := make([]ag.Node, len(testData))
		var y []ag.Node
		for i := 0; i < tt.forwardSteps; i++ {
			g := ag.NewGraph()
			ctx := nn.Context{Graph: g, Mode: nn.Training}
			for j := range data {
				x[j] = g.NewVariable(mat.NewVecDense(data[j]), false)
			}
			y = nn.Reify(ctx, model).(*Model).Forward(x...)
		}

		require.Equal(t, len(x), len(y))

		for i, v := range model.Mean.Value().Data() {
			assert.InDeltaf(t, tt.expectedAvg, v, 1e-1, "Momentum %f Mean %d: expected zero, go %f", tt.momentum, i, v)
		}

		for i, v := range model.StdDev.Value().Data() {
			assert.InDeltaf(t, tt.expectedStdDev, v, 1e-1, "Momentum %f StdDev %d: expected %f, got %f", tt.momentum, i, tt.expectedStdDev, v)
		}
	}
}

func TestModel_Inference(t *testing.T) {

	model := New(3)
	model.Mean = nn.NewParam(mat.NewVecDense([]mat.Float{0.0, 0.0, 1.0}))
	model.StdDev = nn.NewParam(mat.NewVecDense([]mat.Float{1.0, 0.5, 1.0}))
	g := ag.NewGraph()
	ctx := nn.Context{Graph: g, Mode: nn.Inference}
	proc := nn.Reify(ctx, model)
	data := []mat.Float{1.0, 2.0, 3.0}
	x := g.NewVariable(mat.NewVecDense(data), false)
	y := proc.(*Model).Forward(x)
	require.Equal(t, 1, len(y))
	assert.InDeltaSlice(t, []mat.Float{1.0, 4.0, 2.0}, y[0].Value().Data(), 1e-3)
}

func Test_Serialize(t *testing.T) {
	model := NewWithMomentum(3, 0.777)
	model.Mean = nn.NewParam(mat.NewVecDense([]mat.Float{0.0, 0.0, 1.0}))
	model.StdDev = nn.NewParam(mat.NewVecDense([]mat.Float{1.0, 0.5, 1.0}))
	tempFile, err := ioutil.TempFile("", "test_serialize")
	require.Nil(t, err)
	tempFile.Close()
	defer func() {
		_ = os.Remove(tempFile.Name())
	}()
	err = utils.SerializeToFile(tempFile.Name(), &model)
	require.Nil(t, err)

	model2 := New(3)
	err = utils.DeserializeFromFile(tempFile.Name(), &model2)
	require.NoError(t, err)
	require.Equal(t, model.Momentum.Value().Scalar(), model2.Momentum.Value().Scalar())
	require.Equal(t, model.Mean.Value().Data(), model2.Mean.Value().Data())
	require.Equal(t, model.StdDev.Value().Data(), model2.StdDev.Value().Data())
}

func TestModel_Forward(t *testing.T) {

	model := newTestModel()
	g := ag.NewGraph()
	ctx := nn.Context{Graph: g, Mode: nn.Training}

	// == Forward

	x1 := g.NewVariable(mat.NewVecDense([]mat.Float{0.4, 0.8, -0.7, -0.5}), true)
	x2 := g.NewVariable(mat.NewVecDense([]mat.Float{-0.4, -0.6, -0.2, -0.9}), true)
	x3 := g.NewVariable(mat.NewVecDense([]mat.Float{0.4, 0.4, 0.2, 0.8}), true)

	y := rectify(g, nn.Reify(ctx, model).(*Model).Forward(x1, x2, x3)) // TODO: rewrite tests without activation function

	assert.InDeltaSlice(t, []mat.Float{1.1828427, 0.2, 0.0, 0.0}, y[0].Value().Data(), 1.0e-06)
	assert.InDeltaSlice(t, []mat.Float{0.334314, 0.2, 0.0, 0.0}, y[1].Value().Data(), 1.0e-06)
	assert.InDeltaSlice(t, []mat.Float{1.1828427, 0.2, 0.0, 1.302356}, y[2].Value().Data(), 1.0e-06)

	// == Backward

	y[0].PropagateGrad(mat.NewVecDense([]mat.Float{-1.0, -0.2, 0.4, 0.6}))
	y[1].PropagateGrad(mat.NewVecDense([]mat.Float{-0.3, 0.1, 0.7, 0.9}))
	y[2].PropagateGrad(mat.NewVecDense([]mat.Float{0.3, -0.4, 0.7, -0.8}))
	g.BackwardAll()

	assert.InDeltaSlice(t, []mat.Float{-0.6894291116772131, 0.0, 0.0, 0.1265151774227913}, x1.Grad().Data(), 1.0e-06)
	assert.InDeltaSlice(t, []mat.Float{-1.767774815419898e-11, 0.0, 0.0, -0.09674690039596812}, x2.Grad().Data(), 1.0e-06)
	assert.InDeltaSlice(t, []mat.Float{0.6894291116595355, 0.0, 0.0, -0.029768277056219317}, x3.Grad().Data(), 1.0e-06)
	assert.InDeltaSlice(t, []mat.Float{-1.0, -0.5, 0.0, -0.8}, model.B.Grad().Data(), 1.0e-06)
	assert.InDeltaSlice(t, []mat.Float{-0.070710, -0.475556, 0.0, -1.102356}, model.W.Grad().Data(), 1.0e-06)
}

func rectify(g *ag.Graph, xs []ag.Node) []ag.Node {
	ys := make([]ag.Node, len(xs))
	for i, x := range xs {
		ys[i] = g.ReLU(x)
	}
	return ys
}

func newTestModel() *Model {
	model := New(4)
	model.W.Value().SetData([]mat.Float{0.4, 0.0, -0.3, 0.8})
	model.B.Value().SetData([]mat.Float{0.9, 0.2, -0.9, 0.2})
	return model
}
