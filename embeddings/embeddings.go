// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package embeddings

import (
	"bytes"
	"encoding/gob"
	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/embeddings/syncmap"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/nn"
	"github.com/nlpodyssey/spago/utils/kvdb"
	"log"
	"strings"
)

var _ nn.Model[float32] = &Model[float32]{}

var allModels []interface {
	Close()
	ClearUsedEmbeddings()
}

// Model implements an embeddings model.
type Model[T mat.DType] struct {
	nn.BaseModel[T]
	Config
	Storage        *kvdb.KeyValueDB
	UsedEmbeddings *syncmap.Map `spago:"type:params;scope:model"`
	ZeroEmbedding  nn.Param[T]  `spago:"type:weights"`
}

// Config provides configuration settings for an embeddings Model.
type Config struct {
	// Size of the embedding vectors.
	Size int
	// Whether to return the `ZeroEmbedding` in case the word doesn't exist in the embeddings map.
	// If it is false, nil is returned instead, so the caller has more responsibility but also more control.
	UseZeroEmbedding bool
	// The path to DB on the drive
	DBPath string
	// Whether to use the map in read-only mode (embeddings are not updated during training).
	ReadOnly bool
	// Whether to force the deletion of any existing DB to start with an empty embeddings map.
	ForceNewDB bool
}

func init() {
	gob.Register(&Model[float32]{})
	gob.Register(&Model[float64]{})
}

// New returns a new embedding model.
func New[T mat.DType](config Config) *Model[T] {
	m := &Model[T]{
		Config: config,
		Storage: kvdb.NewDefaultKeyValueDB(kvdb.Config{
			Path:     config.DBPath,
			ReadOnly: config.ReadOnly,
			ForceNew: config.ForceNewDB,
		}),
		UsedEmbeddings: syncmap.New(),
		ZeroEmbedding:  nn.NewParam[T](mat.NewEmptyVecDense[T](config.Size), nn.RequiresGrad[T](false)),
	}
	allModels = append(allModels, m)
	return m
}

// Close closes the DB underlying the model of the embeddings map.
// It automatically clears the cache.
func (m *Model[T]) Close() {
	_ = m.Storage.Close() // explicitly ignore errors here
	m.ClearUsedEmbeddings()
}

// ClearUsedEmbeddings clears the cache of the used embeddings.
// Beware of any external references to the values of m.UsedEmbeddings. These are weak references!
func (m *Model[T]) ClearUsedEmbeddings() {
	m.UsedEmbeddings.Range(func(key interface{}, value interface{}) bool {
		m.UsedEmbeddings.Delete(key)
		return true
	})
}

// DropAll clears the cache of used embeddings and drops all the data stored in the DB.
func (m *Model[T]) DropAll() error {
	m.ClearUsedEmbeddings()
	return m.Storage.DropAll()
}

// Close closes the DBs underlying all instantiated embeddings models.
// It automatically clears the caches.
func Close() {
	for _, model := range allModels {
		model.Close()
	}
}

// ClearUsedEmbeddings clears the cache of the used embeddings of all instantiated embeddings models.
// Beware of any external references to the values of m.UsedEmbeddings. These are weak references!
func ClearUsedEmbeddings() {
	for _, model := range allModels {
		model.ClearUsedEmbeddings()
	}
}

// Count counts how many embeddings are stored in the DB.
// It invokes log.Fatal in case of reading errors.
func (m *Model[T]) Count() int {
	keys, err := m.Storage.Keys()
	if err != nil {
		log.Fatal(err)
	}
	return len(keys)
}

// SetEmbedding inserts a new word embedding.
// If the word is already on the map, it overwrites the existing value with the new one.
func (m *Model[T]) SetEmbedding(word string, value mat.Matrix[T]) {
	if m.ReadOnly {
		log.Fatal("embedding: set operation not permitted in read-only mode")
	}

	embedding := nn.NewParam[T](value)
	embedding.SetPayload(nn.NewPayload[T]())

	buf := new(bytes.Buffer)
	err := nn.MarshalBinaryParam(embedding, buf)
	if err != nil {
		log.Fatal(err)
	}

	err = m.Storage.Put([]byte(word), buf.Bytes())
	if err != nil {
		log.Fatal(err)
	}
}

// SetEmbeddingFromData inserts a new word embeddings.
// If the word is already on the map, overwrites the existing value with the new one.
func (m *Model[T]) SetEmbeddingFromData(word string, data []T) {
	vec := mat.NewVecDense(data)
	defer mat.ReleaseDense(vec)
	m.SetEmbedding(word, vec)
}

// GetStoredEmbedding returns the parameter (the word embedding) associated with the given word.
// It first looks for the exact correspondence of the word. If there is no match, it tries the word lowercase.
//
// The returned embedding is also cached in m.UsedEmbeddings for two reasons:
//     - to allow a faster recovery;
//     - to keep track of used embeddings, should they be optimized.
//
// If no embedding is found, nil is returned.
// It panics in case of Storage errors.
func (m *Model[T]) GetStoredEmbedding(word string) nn.Param[T] {
	if found := m.getStoredEmbedding(word); found != nil {
		return found
	}
	if found := m.getStoredEmbedding(strings.ToLower(word)); found != nil {
		return found
	}
	return nil
}

// getStoredEmbedding returns the parameter (the word embedding) associated with the given word (exact correspondence).
// The returned embedding is also cached in m.UsedEmbeddings for two reasons:
//     - to allow a faster recovery;
//     - to keep track of used embeddings, should they be optimized.
// It panics in case of Storage errors.
func (m *Model[T]) getStoredEmbedding(word string) nn.Param[T] {
	if embedding, ok := m.getUsedEmbedding(word); ok {
		return embedding
	}
	data, ok, err := m.Storage.Get([]byte(word))
	if err != nil {
		log.Fatal(err)
	}
	if !ok {
		return nil // embedding not found
	}

	embedding := nn.NewParam[T](nil, nn.SetStorage[T](m.Storage), nn.RequiresGrad[T](!m.ReadOnly))
	embedding.SetName(word)
	err = nn.UnmarshalBinaryParamWithReceiver(bytes.NewReader(data), embedding)
	if err != nil {
		log.Fatal(err)
	}

	m.UsedEmbeddings.Store(word, embedding) // important
	return embedding
}

func (m *Model[T]) getUsedEmbedding(word string) (nn.Param[T], bool) {
	if value, ok := m.UsedEmbeddings.Load(word); ok {
		return value.(nn.Param[T]), true
	}
	return nil, false
}

// Encode returns the embeddings associated with the input words.
// The embeddings are returned as Node(s) already inserted in the graph.
// To words that have no embeddings, the corresponding nodes
// are nil or the `ZeroEmbedding`, depending on the configuration.
func (m *Model[T]) Encode(words []string) []ag.Node[T] {
	encoding := make([]ag.Node[T], len(words))
	cache := make(map[string]ag.Node[T]) // be smart, don't create two nodes for the same word!
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

// getStoredEmbedding returns the embedding associated to the word.
// If no embedding is found, nil or the `ZeroEmbedding` is returned, depending on the model configuration.
func (m *Model[T]) getEmbedding(word string) ag.Node[T] {
	switch param := m.GetStoredEmbedding(word); {
	case param == nil:
		if m.Config.UseZeroEmbedding {
			return m.ZeroEmbedding
		}
		return nil
	default:
		return m.Graph().NewWrap(param)
	}
}
