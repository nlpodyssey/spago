// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package emb

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/optimizers/gd"
	"log"
)

type Embedding struct {
	*nn.Param
	storage *Map
}

// ApplyDelta applies the dalta and updates the entry of the underlying storage.
// It panics if the storage is read-only.
func (e *Embedding) ApplyDelta(delta mat.Matrix) {
	if e.storage.ReadOnly {
		log.Fatal("emb: read-only embeddings cannot apply delta")
	}
	e.Param.ApplyDelta(delta)
	if _, err := e.storage.update(e); err != nil {
		log.Fatal(err)
	}
}

// TrackEmbeddings tells the optimizer to track the embeddings.
func TrackEmbeddings(ex []*Embedding, o *gd.GradientDescent) {
	for _, e := range ex {
		o.Track(e)
	}
}

// UntrackEmbeddings tells the optimizer to untrack the embeddings.
func UntrackEmbeddings(ex []*Embedding, o *gd.GradientDescent) {
	for _, e := range ex {
		o.Untrack(e)
	}
}
