// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package evolvingembeddings

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/stretchr/testify/assert"
	"testing"
)

// TODO: split test in multiple functions; increase tests coverage
func TestModel_NewAggregateDropAll(t *testing.T) {
	t.Run("float32", testModelNewAggregateDropAll[float32])
	t.Run("float64", testModelNewAggregateDropAll[float64])
}

func testModelNewAggregateDropAll[T mat.DType](t *testing.T) {
	model := New[T](Config{
		Size:             12,
		PoolingOperation: Min,
		DBPath:           "/tmp/evolvingembeddings_test",
		ForceNewDB:       true,
	})
	wordInContext1 := &WordVectorPair[T]{
		Word:   "foo",
		Vector: mat.NewVecDense([]T{0.1, 0.2, 0.3, 0.4, 0.5, -0.6, -0.5, 0.8, -0.8, -3, -0.3, -0.4}),
	}
	sameWordInContext2 := &WordVectorPair[T]{
		Word:   "foo",
		Vector: mat.NewVecDense([]T{0.2, 0.7, 0.5, 0.0, 0.4, 0.5, -0.8, 0.7, -0.3, 0.2, -0.0, -0.9}),
	}

	g := ag.NewGraph[T]()
	proc := nn.ReifyForTraining(model, g)
	res := proc.Encode([]string{"foo"})[0]
	assert.InDeltaSlice(t, wordInContext1.Vector.ZerosLike().Data(), res.Value().Data(), 1.0e-6)

	model.Aggregate([]*WordVectorPair[T]{wordInContext1})

	g = ag.NewGraph[T]()
	proc = nn.ReifyForTraining(model, g)
	res = proc.Encode([]string{"foo"})[0]
	assert.InDeltaSlice(t, wordInContext1.Vector.Data(), res.Value().Data(), 1.0e-6)

	model.Aggregate([]*WordVectorPair[T]{sameWordInContext2})

	g = ag.NewGraph[T]()
	proc = nn.ReifyForTraining(model, g)
	res = proc.Encode([]string{"foo"})[0]
	assert.InDeltaSlice(t, []T{
		0.1, 0.2, 0.3, 0.0, 0.4, -0.6, -0.8, 0.7, -0.8, -3, -0.3, -0.9,
	}, res.Value().Data(), 1.0e-6)

	err := model.DropAll()
	if err != nil {
		t.Error(err)
	}

	g = ag.NewGraph[T]()
	proc = nn.ReifyForTraining(model, g)
	res = proc.Encode([]string{"foo"})[0]
	assert.InDeltaSlice(t, []T{
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
	}, res.Value().Data(), 1.0e-6)
}
