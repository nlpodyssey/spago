// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// A word embedding model that evolves itself by dynamically aggregating contextual embeddings over time during inference.
// See "Pooled Contextualized Embeddings" by Akbik et al., 2019 https://www.aclweb.org/anthology/papers/N/N19/N19-1078/
package evolvingembeddings

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/nlp/embeddings"
)

var (
	_ nn.Model     = &Model{}
	_ nn.Processor = &Processor{}
)

var allModels []*Model

type Model struct {
	Config
	delegate *embeddings.Model
}

type PoolingType int

const (
	Max PoolingType = iota
	Min
	// TODO: Avg
)

type Config struct {
	// Size of the embedding vectors.
	Size int
	// The operation used to aggregate new embeddings with pre-existing ones.
	PoolingOperation PoolingType
	// The path to DB on the drive
	DBPath string
	// Whether to force the deletion of any existing DB to start with an empty embeddings map.
	ForceNewDB bool
}

// New returns a new embedding model.
func New(config Config) *Model {
	m := &Model{
		Config: config,
		delegate: embeddings.New(embeddings.Config{
			Size:             config.Size,
			UseZeroEmbedding: true,
			DBPath:           config.DBPath,
			ReadOnly:         false,
			ForceNewDB:       config.ForceNewDB,
		}),
	}
	allModels = append(allModels, m)
	return m
}

// Close closes the DB underlying the model of the embeddings map.
func (m *Model) Close() {
	m.delegate.Close()
	m.delegate.ClearUsedEmbeddings()
}

func (m *Model) DropAll() error {
	return m.delegate.DropAll()
}

// Close closes the DBs underlying all instantiated embeddings models.
// It automatically clears the caches.
func Close() {
	for _, model := range allModels {
		model.Close()
	}
}

func (m *Model) Count() int {
	return m.delegate.Count()
}

type WordVectorPair struct {
	Word   string
	Vector *mat.Dense
}

func (m *Model) Aggregate(list []*WordVectorPair) {
	m.delegate.ClearUsedEmbeddings()
	for _, item := range list {
		m.aggregate(item.Word, item.Vector)
	}
}

func (m *Model) aggregate(word string, vector *mat.Dense) {
	defer m.delegate.ClearUsedEmbeddings() // important
	if found := m.delegate.GetEmbedding(word); found == nil {
		m.delegate.SetEmbedding(word, vector)
	} else {
		m.delegate.SetEmbedding(word, m.pooling(found.Value().(*mat.Dense), vector))
	}
}

func (m *Model) pooling(a, b *mat.Dense) *mat.Dense {
	switch m.PoolingOperation {
	case Max:
		return a.Maximum(b)
	case Min:
		return a.Minimum(b)
	default:
		panic("evolvingembeddings: invalid pooling operation")
	}
}

type Processor struct {
	nn.BaseProcessor
	delegate *embeddings.Processor
}

func (m *Model) NewProc(g *ag.Graph) nn.Processor {
	return &Processor{
		BaseProcessor: nn.BaseProcessor{
			Model:             m,
			Mode:              nn.Training,
			Graph:             g,
			FullSeqProcessing: false,
		},
		delegate: m.delegate.NewProc(g).(*embeddings.Processor),
	}
}

// Encodes returns the embeddings associated with the input words.
func (p *Processor) Encode(words []string) []ag.Node {
	encoding := make([]ag.Node, len(words))
	cache := make(map[string]ag.Node) // be smart, don't create two nodes for the same word!
	for i, word := range words {
		if item, ok := cache[word]; ok {
			encoding[i] = item
		} else {
			embedding := p.getEmbedding(word)
			encoding[i], cache[word] = embedding, embedding
		}
	}
	return encoding
}

func (p *Processor) getEmbedding(words string) ag.Node {
	model := p.Model.(*Model)
	switch param := model.delegate.GetEmbedding(words); {
	case param == nil:
		return p.Graph.NewWrapNoGrad(p.delegate.ZeroEmbedding) // must not be nil; important no grad
	default:
		return p.Graph.NewWrapNoGrad(param) // important no grad
	}
}

func (p *Processor) Forward(_ ...ag.Node) []ag.Node {
	panic("embeddings: p.Forward() not implemented. Use p.Encode() instead.")
}
