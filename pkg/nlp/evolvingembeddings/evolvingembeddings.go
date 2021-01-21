// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package evolvingembeddings provides a word embedding model that evolves by dynamically
// aggregating contextual embeddings over time during inference.
// See "Pooled Contextualized Embeddings" by Akbik et al., 2019
// https://www.aclweb.org/anthology/papers/N/N19/N19-1078/
package evolvingembeddings

import (
	"bytes"
	"encoding/gob"
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/utils/kvdb"
	"log"
	"strings"
	"sync"
)

var (
	_ nn.Model = &Model{}
)

var allModels []*Model

// Model implements an Evolving Pooled Contextualized Embeddings model.
type Model struct {
	nn.BaseModel
	Config
	Storage       *kvdb.KeyValueDB
	Mu            sync.Mutex
	ZeroEmbedding nn.Param `spago:"type:weights"`
}

// PoolingType is the enumeration-like type used to distinguish different types
// of pooling operations for an Evolving Pooled Contextualized Embeddings Model.
type PoolingType int

const (
	// Max identifies the maximum pooling operation function.
	Max PoolingType = iota
	// Min identifies the minimum pooling operation function.
	Min
	// TODO: Avg
)

// Config provides configuration settings for an Evolving Pooled Contextualized Embeddings Model.
type Config struct {
	// Size of the embedding vectors.
	Size int
	// The operation used to aggregate new embeddings with pre-existing ones.
	PoolingOperation PoolingType
	// The path to DB on the drive
	DBPath string
	// Whether to force the deletion of any existing DB to start with an empty embeddings mam.
	ForceNewDB bool
}

func init() {
	gob.Register(&Model{})
}

// New returns a new embedding Model.
func New(config Config) *Model {
	m := &Model{
		Config: config,
		Storage: kvdb.NewDefaultKeyValueDB(kvdb.Config{
			Path:     config.DBPath,
			ReadOnly: false,
			ForceNew: config.ForceNewDB,
		}),
		ZeroEmbedding: nn.NewParam(mat.NewEmptyVecDense(config.Size), nn.RequiresGrad(false)),
	}
	allModels = append(allModels, m)
	return m
}

// Close closes the DB underlying the model of the embeddings mam.
func (m *Model) Close() {
	_ = m.Storage.Close() // explicitly ignore errors here
}

// DropAll drops all the data stored in the DB.
func (m *Model) DropAll() error {
	return m.Storage.DropAll()
}

// Close closes the DBs underlying all instantiated embeddings models.
func Close() {
	for _, model := range allModels {
		model.Close()
	}
}

// Count counts how many embeddings are stored in the DB.
// It invokes log.Fatal in case of reading errors.
func (m *Model) Count() int {
	keys, err := m.Storage.Keys()
	if err != nil {
		log.Fatal(err)
	}
	return len(keys)
}

// WordVectorPair associates a Vector to a Word.
type WordVectorPair struct {
	Word   string
	Vector mat.Matrix
}

// Aggregate performs a pooling operation over the list of WordVectorPair elements.
func (m *Model) Aggregate(list []*WordVectorPair) {
	for _, item := range list {
		word := item.Word
		vector := item.Vector
		if found := m.getStorageEmbedding(word); found == nil {
			m.setEmbedding(word, vector)
		} else {
			m.setEmbedding(word, m.pooling(found, vector))
		}
	}
}

func (m *Model) pooling(a, b mat.Matrix) mat.Matrix {
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
func (m *Model) setEmbedding(word string, value mat.Matrix) {
	var err error
	var data []byte = nil

	if value != nil {
		buf := new(bytes.Buffer)
		err = mat.MarshalBinaryMatrix(value, buf)
		if err != nil {
			log.Fatal(err)
		}
		data = buf.Bytes()
	}

	if err := m.Storage.Put([]byte(word), data); err != nil {
		log.Fatal(err)
	}
}

// getEmbedding returns the vector (the word embedding) associated with the given word.
// It first looks for the exact correspondence of the word. If there is no match, it tries the word lowercase.
// If no embedding is found, nil is returned.
// It panics in case of Storage errors.
func (m *Model) getStorageEmbedding(word string) mat.Matrix {
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
// It panics in case of Storage errors.
func (m *Model) getEmbeddingExactMatch(word string) mat.Matrix {
	data, ok, err := m.Storage.Get([]byte(word))
	if err != nil {
		log.Fatal(err)
	}
	if !ok || data == nil {
		return nil // embedding not found, or nil Dense matrix
	}

	embedding, err := mat.UnmarshalBinaryMatrix(bytes.NewReader(data))
	if err != nil {
		log.Fatal(err)
	}
	return embedding
}

// Encode returns the embeddings associated with the input words.
func (m *Model) Encode(words []string) []ag.Node {
	encoding := make([]ag.Node, len(words))
	cache := make(map[string]ag.Node) // be smart, don't create two nodes for the same word!
	for i, word := range words {
		if item, ok := cache[word]; ok {
			encoding[i] = item
		} else {
			embedding := m.getEmbedding(word)
			encoding[i], cache[word] = embedding, embedding
		}
	}
	return encoding
}

func (m *Model) getEmbedding(words string) ag.Node {
	switch vector := m.getStorageEmbedding(words); {
	case vector == nil:
		return m.ZeroEmbedding // must not be nil; important no grad
	default:
		return m.Graph().NewVariable(vector, false)
	}
}
