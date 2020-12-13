// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package evolvingembeddings provides s word embedding model that evolves by dynamically
// aggregating contextual embeddings over time during inference.
// See "Pooled Contextualized Embeddings" by Akbik et al., 2019
// https://www.aclweb.org/anthology/papers/N/N19/N19-1078/
package evolvingembeddings

import (
	"bytes"
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/utils/kvdb"
	"log"
	"strings"
	"sync"
)

var (
	_ nn.Model     = &Model{}
	_ nn.Processor = &Processor{}
)

var allModels []*Model

type Model struct {
	Config
	storage       kvdb.KeyValueDB
	mu            sync.Mutex
	ZeroEmbedding *nn.Param `type:"weights"`
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
		storage: kvdb.NewDefaultKeyValueDB(kvdb.Config{
			Path:     config.DBPath,
			ReadOnly: false,
			ForceNew: config.ForceNewDB,
		}),
		ZeroEmbedding: nn.NewParam(mat.NewEmptyVecDense(config.Size)),
	}
	nn.RequiresGrad(false)(m.ZeroEmbedding)
	allModels = append(allModels, m)
	return m
}

// Close closes the DB underlying the model of the embeddings map.
func (m *Model) Close() {
	_ = m.storage.Close() // explicitly ignore errors here
}

func (m *Model) DropAll() error {
	return m.storage.DropAll()
}

// Close closes the DBs underlying all instantiated embeddings models.
func Close() {
	for _, model := range allModels {
		model.Close()
	}
}

func (m *Model) Count() int {
	keys, err := m.storage.Keys()
	if err != nil {
		log.Fatal(err)
	}
	return len(keys)
}

type WordVectorPair struct {
	Word   string
	Vector *mat.Dense
}

func (m *Model) Aggregate(list []*WordVectorPair) {
	for _, item := range list {
		word := item.Word
		vector := item.Vector
		if found := m.getEmbedding(word); found == nil {
			m.setEmbedding(word, vector)
		} else {
			m.setEmbedding(word, m.pooling(found, vector))
		}
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

// SetEmbeddings inserts a new word embeddings.
// If the word is already on the map, overwrites the existing value with the new one.
func (m *Model) setEmbedding(word string, value *mat.Dense) {
	var buf bytes.Buffer
	if _, err := mat.MarshalBinaryTo(value, &buf); err != nil {
		log.Fatal(err)
	}
	if err := m.storage.Put([]byte(word), buf.Bytes()); err != nil {
		log.Fatal(err)
	}
}

// getEmbedding returns the vector (the word embedding) associated with the given word.
// It first looks for the exact correspondence of the word. If there is no match, it tries the word lowercase.
// If no embedding is found, nil is returned.
// It panics in case of storage errors.
func (m *Model) getEmbedding(word string) *mat.Dense {
	if found := m.getEmbeddingExactMatch(word); found != nil {
		return found
	}
	if found := m.getEmbeddingExactMatch(strings.ToLower(word)); found != nil {
		return found
	}
	return nil
}

// getEmbeddingExactMatch returns the vector (the word embedding) associated with the given word (exact correspondence).
// If no embedding is found, nil is returned.
// It panics in case of storage errors.
func (m *Model) getEmbeddingExactMatch(word string) *mat.Dense {
	data, ok, err := m.storage.Get([]byte(word))
	if err != nil {
		log.Fatal(err)
	}
	if !ok {
		return nil // embedding not found
	}
	embedding, _, err := mat.NewUnmarshalBinaryFrom(bytes.NewReader(data))
	if err != nil {
		log.Fatal(err)
	}
	return embedding
}

type Processor struct {
	nn.BaseProcessor
	ZeroEmbedding ag.Node
}

func (m *Model) NewProc(ctx nn.Context) nn.Processor {
	return &Processor{
		BaseProcessor: nn.BaseProcessor{
			Model:             m,
			Mode:              ctx.Mode,
			Graph:             ctx.Graph,
			FullSeqProcessing: false,
		},
		ZeroEmbedding: ag.NewWrapNoGrad(m.ZeroEmbedding),
	}
}

// Encode returns the embeddings associated with the input words.
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
	switch vector := model.getEmbedding(words); {
	case vector == nil:
		return p.Graph.NewWrapNoGrad(p.ZeroEmbedding) // must not be nil; important no grad
	default:
		return p.Graph.NewVariable(vector, false)
	}
}

func (p *Processor) Forward(_ ...ag.Node) []ag.Node {
	panic("embeddings: p.Forward() not implemented. Use p.Encode() instead.")
}
