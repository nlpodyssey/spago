// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package batchnorm

import (
	"github.com/stretchr/testify/assert"
	"golang.org/x/exp/rand"
	"os"
	"testing"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/nn"
	"github.com/nlpodyssey/spago/utils"
	"github.com/stretchr/testify/require"
)

func TestModel_Forward_Params(t *testing.T) {
	t.Run("float32", testModelForwardParams[float32])
	t.Run("float64", testModelForwardParams[float64])
}

func testModelForwardParams[T mat.DType](t *testing.T) {
	const numDataInstances = 10000
	const dataSize = 100

	testCases := []struct {
		shift          T
		multiplier     T
		momentum       T
		expectedAvg    T
		expectedStdDev T
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

	testData := make([][]T, numDataInstances)
	for i := range testData {
		testData[i] = make([]T, dataSize)
		for j := range testData[i] {
			testData[i][j] = T(rnd.NormFloat64())
		}
	}

	for _, tt := range testCases {
		model := NewWithMomentum(dataSize, tt.momentum)
		data := make([][]T, len(testData))
		for i := range testData {
			data[i] = make([]T, dataSize)
			for j := range testData[i] {
				data[i][j] = tt.multiplier*testData[i][j] + tt.shift
			}
		}

		x := make([]ag.Node[T], len(testData))
		var y []ag.Node[T]
		for i := 0; i < tt.forwardSteps; i++ {
			s := ag.NewSession[T](model, ag.Training)
			for j := range data {
				x[j] = s.NewVariable(mat.NewVecDense(data[j]), false)
			}
			y = s.Module().Forward(x...)
			s.Close()
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
	t.Run("float32", testModelInference[float32])
	t.Run("float64", testModelInference[float64])
}

func testModelInference[T mat.DType](t *testing.T) {

	model := New[T](3)
	model.Mean = nn.NewParam[T](mat.NewVecDense[T]([]T{0.0, 0.0, 1.0}))
	model.StdDev = nn.NewParam[T](mat.NewVecDense[T]([]T{1.0, 0.5, 1.0}))
	model.W = nn.NewParam[T](mat.NewInitVecDense[T](3, 1.0))

	s := ag.NewSession[T](model, ag.Inference)
	defer s.Close()

	data := []T{1.0, 2.0, 3.0}
	x := s.NewVariable(mat.NewVecDense[T](data), false)
	y := s.Module().Forward(x)
	require.Equal(t, 1, len(y))
	assert.InDeltaSlice(t, []T{1.0, 4.0, 2.0}, y[0].Value().Data(), 1e-3)
}

func Test_Serialize(t *testing.T) {
	t.Run("float32", testSerialize[float32])
	t.Run("float64", testSerialize[float64])
}

func testSerialize[T mat.DType](t *testing.T) {
	model := NewWithMomentum[T](3, 0.777)
	model.Mean = nn.NewParam[T](mat.NewVecDense[T]([]T{0.0, 0.0, 1.0}))
	model.StdDev = nn.NewParam[T](mat.NewVecDense[T]([]T{1.0, 0.5, 1.0}))
	tempFile, err := os.CreateTemp("", "test_serialize")
	require.Nil(t, err)
	tempFile.Close()
	defer func() {
		_ = os.Remove(tempFile.Name())
	}()
	err = utils.SerializeToFile(tempFile.Name(), &model)
	require.Nil(t, err)

	model2 := New[T](3)
	err = utils.DeserializeFromFile(tempFile.Name(), &model2)
	require.NoError(t, err)
	require.Equal(t, model.Momentum.Value().Scalar(), model2.Momentum.Value().Scalar())
	require.Equal(t, model.Mean.Value().Data(), model2.Mean.Value().Data())
	require.Equal(t, model.StdDev.Value().Data(), model2.StdDev.Value().Data())
}

func TestModel_Forward(t *testing.T) {
	t.Run("float32", testModelForward[float32])
	t.Run("float64", testModelForward[float64])
}

func testModelForward[T mat.DType](t *testing.T) {
	model := newTestModel[T]()
	s := ag.NewSession[T](model, ag.Training)
	defer s.Close()

	// == Forward

	x1 := s.NewVariable(mat.NewVecDense[T]([]T{0.4, 0.8, -0.7, -0.5}), true)
	x2 := s.NewVariable(mat.NewVecDense[T]([]T{-0.4, -0.6, -0.2, -0.9}), true)
	x3 := s.NewVariable(mat.NewVecDense[T]([]T{0.4, 0.4, 0.2, 0.8}), true)

	y := rectify(s.Module().Forward(x1, x2, x3)) // TODO: rewrite tests without activation function

	assert.InDeltaSlice(t, []T{1.1828427, 0.2, 0.0, 0.0}, y[0].Value().Data(), 1.0e-04)
	assert.InDeltaSlice(t, []T{0.334314, 0.2, 0.0, 0.0}, y[1].Value().Data(), 1.0e-04)
	assert.InDeltaSlice(t, []T{1.1828427, 0.2, 0.0, 1.302356}, y[2].Value().Data(), 1.0e-04)

	// == Backward

	y[0].PropagateGrad(mat.NewVecDense[T]([]T{-1.0, -0.2, 0.4, 0.6}))
	y[1].PropagateGrad(mat.NewVecDense[T]([]T{-0.3, 0.1, 0.7, 0.9}))
	y[2].PropagateGrad(mat.NewVecDense[T]([]T{0.3, -0.4, 0.7, -0.8}))
	s.Graph().Backward(nil)

	assert.InDeltaSlice(t, []T{-0.6894291116772131, 0.0, 0.0, 0.1265151774227913}, x1.Grad().Data(), 1.0e-04)
	assert.InDeltaSlice(t, []T{-1.767774815419898e-11, 0.0, 0.0, -0.09674690039596812}, x2.Grad().Data(), 1.0e-04)
	assert.InDeltaSlice(t, []T{0.6894291116595355, 0.0, 0.0, -0.029768277056219317}, x3.Grad().Data(), 1.0e-04)
	assert.InDeltaSlice(t, []T{-1.0, -0.5, 0.0, -0.8}, model.B.Grad().Data(), 1.0e-04)
	assert.InDeltaSlice(t, []T{-0.070710, -0.475556, 0.0, -1.102356}, model.W.Grad().Data(), 1.0e-04)
}

func rectify[T mat.DType](xs []ag.Node[T]) []ag.Node[T] {
	ys := make([]ag.Node[T], len(xs))
	for i, x := range xs {
		ys[i] = ag.ReLU(x)
	}
	return ys
}

func newTestModel[T mat.DType]() *Model[T] {
	model := New[T](4)
	model.W.Value().SetData([]T{0.4, 0.0, -0.3, 0.8})
	model.B.Value().SetData([]T{0.9, 0.2, -0.9, 0.2})
	return model
}
