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
	model := New[mat.Float](Config{
		Size:             12,
		PoolingOperation: Min,
		DBPath:           "/tmp/evolvingembeddings_test",
		ForceNewDB:       true,
	})
	wordInContext1 := &WordVectorPair[mat.Float]{
		Word:   "foo",
		Vector: mat.NewVecDense([]mat.Float{0.1, 0.2, 0.3, 0.4, 0.5, -0.6, -0.5, 0.8, -0.8, -3, -0.3, -0.4}),
	}
	sameWordInContext2 := &WordVectorPair[mat.Float]{
		Word:   "foo",
		Vector: mat.NewVecDense([]mat.Float{0.2, 0.7, 0.5, 0.0, 0.4, 0.5, -0.8, 0.7, -0.3, 0.2, -0.0, -0.9}),
	}

	g := ag.NewGraph[mat.Float]()
	proc := nn.ReifyForTraining(model, g)
	res := proc.Encode([]string{"foo"})[0]
	assert.InDeltaSlice(t, wordInContext1.Vector.ZerosLike().Data(), res.Value().Data(), 1.0e-6)

	model.Aggregate([]*WordVectorPair[mat.Float]{wordInContext1})

	g = ag.NewGraph[mat.Float]()
	proc = nn.ReifyForTraining(model, g)
	res = proc.Encode([]string{"foo"})[0]
	assert.InDeltaSlice(t, wordInContext1.Vector.Data(), res.Value().Data(), 1.0e-6)

	model.Aggregate([]*WordVectorPair[mat.Float]{sameWordInContext2})

	g = ag.NewGraph[mat.Float]()
	proc = nn.ReifyForTraining(model, g)
	res = proc.Encode([]string{"foo"})[0]
	assert.InDeltaSlice(t, []mat.Float{
		0.1, 0.2, 0.3, 0.0, 0.4, -0.6, -0.8, 0.7, -0.8, -3, -0.3, -0.9,
	}, res.Value().Data(), 1.0e-6)

	err := model.DropAll()
	if err != nil {
		t.Error(err)
	}

	g = ag.NewGraph[mat.Float]()
	proc = nn.ReifyForTraining(model, g)
	res = proc.Encode([]string{"foo"})[0]
	assert.InDeltaSlice(t, []mat.Float{
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
	}, res.Value().Data(), 1.0e-6)
}
