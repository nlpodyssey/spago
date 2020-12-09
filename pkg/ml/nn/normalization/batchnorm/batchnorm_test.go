// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package batchnorm

import (
	"io/ioutil"
	"math/rand"
	"os"
	"testing"

	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/utils"
	"github.com/stretchr/testify/require"
	"gonum.org/v1/gonum/floats"
)

func TestModel_Forward_Params(t *testing.T) {
	const numDataInstances = 10000
	const dataSize = 100

	testCases := []struct {
		shift          float64
		multiplier     float64
		momentum       float64
		expectedAvg    float64
		expectedStdDev float64
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

	testData := make([][]float64, numDataInstances)
	for i := range testData {
		testData[i] = make([]float64, dataSize)
		for j := range testData[i] {
			testData[i][j] = rnd.NormFloat64()
		}
	}

	for _, tt := range testCases {
		model := NewWithMomentum(dataSize, tt.momentum)
		data := make([][]float64, len(testData))
		for i := range testData {
			data[i] = make([]float64, dataSize)
			for j := range testData[i] {
				data[i][j] = tt.multiplier*testData[i][j] + tt.shift
			}
		}

		x := make([]ag.Node, len(testData))
		var y []ag.Node
		for i := 0; i < tt.forwardSteps; i++ {
			g := ag.NewGraph()
			for j := range data {
				x[j] = g.NewVariable(mat.NewVecDense(data[j]), false)
			}
			y = model.NewProc(nn.Context{Graph: g, Mode: nn.Training}).Forward(x...)
		}

		require.Equal(t, len(x), len(y))

		for i, v := range model.Mean.Value().Data() {
			if !floats.EqualWithinAbs(v, tt.expectedAvg, 1e-1) {
				t.Fatalf("Momentum %f Mean %d: expected zero, go %f", tt.momentum, i, v)
			}
		}

		for i, v := range model.StdDev.Value().Data() {
			if !floats.EqualWithinAbs(v, tt.expectedStdDev, 1e-1) {
				t.Fatalf("Momentum %f StdDev %d: expected %f, got %f", tt.momentum, i, tt.expectedStdDev, v)
			}
		}
	}

}

func TestModel_Inference(t *testing.T) {

	model := New(3)
	model.Mean = nn.NewParam(mat.NewVecDense([]float64{0.0, 0.0, 1.0}))
	model.StdDev = nn.NewParam(mat.NewVecDense([]float64{1.0, 0.5, 1.0}))
	g := ag.NewGraph()
	proc := model.NewProc(nn.Context{Graph: g, Mode: nn.Inference})
	data := []float64{1.0, 2.0, 3.0}
	x := g.NewVariable(mat.NewVecDense(data), false)
	y := proc.Forward(x)
	require.Equal(t, 1, len(y))

	require.True(t, floats.EqualApprox(y[0].Value().Data(), []float64{1.0, 4.0, 2.0}, 1e-3))

}

func Test_Serialize(t *testing.T) {
	model := NewWithMomentum(3, 0.777)
	model.Mean = nn.NewParam(mat.NewVecDense([]float64{0.0, 0.0, 1.0}))
	model.StdDev = nn.NewParam(mat.NewVecDense([]float64{1.0, 0.5, 1.0}))
	tempFile, err := ioutil.TempFile("", "test_serialize")
	require.Nil(t, err)
	tempFile.Close()
	defer func() {
		_ = os.Remove(tempFile.Name())
	}()
	err = utils.SerializeToFile(tempFile.Name(), nn.NewParamsSerializer(model))
	require.Nil(t, err)

	model2 := New(3)
	serializer := nn.NewParamsSerializer(model2)
	tempFile, err = os.Open(tempFile.Name())
	require.Nil(t, err)
	_, err = serializer.Deserialize(tempFile)
	require.NoError(t, err)
	require.Equal(t, model.Momentum.Value().Scalar(), model2.Momentum.Value().Scalar())
	require.Equal(t, model.Mean.Value().Data(), model2.Mean.Value().Data())
	require.Equal(t, model.StdDev.Value().Data(), model2.StdDev.Value().Data())
}
func TestModel_Forward(t *testing.T) {

	model := newTestModel()
	g := ag.NewGraph()

	// == Forward

	x1 := g.NewVariable(mat.NewVecDense([]float64{0.4, 0.8, -0.7, -0.5}), true)
	x2 := g.NewVariable(mat.NewVecDense([]float64{-0.4, -0.6, -0.2, -0.9}), true)
	x3 := g.NewVariable(mat.NewVecDense([]float64{0.4, 0.4, 0.2, 0.8}), true)

	y := rectify(g, model.NewProc(nn.Context{Graph: g, Mode: nn.Training}).Forward(x1, x2, x3)) // TODO: rewrite tests without activation function

	if !floats.EqualApprox(y[0].Value().Data(), []float64{1.1828427, 0.2, 0.0, 0.0}, 1.0e-06) {
		t.Error("The output at position 0 doesn't match the expected values")
	}

	if !floats.EqualApprox(y[1].Value().Data(), []float64{0.334314, 0.2, 0.0, 0.0}, 1.0e-06) {
		t.Error("The output at position 1 doesn't match the expected values")
	}

	if !floats.EqualApprox(y[2].Value().Data(), []float64{1.1828427, 0.2, 0.0, 1.302356}, 1.0e-06) {
		t.Error("The output at position 2 doesn't match the expected values")
	}

	// == Backward

	y[0].PropagateGrad(mat.NewVecDense([]float64{-1.0, -0.2, 0.4, 0.6}))
	y[1].PropagateGrad(mat.NewVecDense([]float64{-0.3, 0.1, 0.7, 0.9}))
	y[2].PropagateGrad(mat.NewVecDense([]float64{0.3, -0.4, 0.7, -0.8}))
	g.BackwardAll()

	if !floats.EqualApprox(x1.Grad().Data(), []float64{-0.6894291116772131, 0.0, 0.0, 0.1265151774227913}, 1.0e-06) {
		t.Error("The x1-gradients don't match the expected values")
	}

	if !floats.EqualApprox(x2.Grad().Data(), []float64{-1.767774815419898e-11, 0.0, 0.0, -0.09674690039596812}, 1.0e-06) {
		t.Error("The x2-gradients don't match the expected values")
	}

	if !floats.EqualApprox(x3.Grad().Data(), []float64{0.6894291116595355, 0.0, 0.0, -0.029768277056219317}, 1.0e-06) {
		t.Error("The x3-gradients don't match the expected values")
	}

	if !floats.EqualApprox(model.B.Grad().Data(), []float64{-1.0, -0.5, 0.0, -0.8}, 1.0e-06) {
		t.Error("The biases B doesn't match the expected values")
	}

	if !floats.EqualApprox(model.W.Grad().Data(), []float64{-0.070710, -0.475556, 0.0, -1.102356}, 1.0e-06) {
		t.Error("The weights W doesn't match the expected values")
	}
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
	model.W.Value().SetData([]float64{0.4, 0.0, -0.3, 0.8})
	model.B.Value().SetData([]float64{0.9, 0.2, -0.9, 0.2})
	return model
}
