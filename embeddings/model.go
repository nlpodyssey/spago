// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package embeddings

import (
	"encoding/gob"
	"fmt"
	"sync"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/embeddings/store"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/nn"
)

// Config provides configuration settings for an embeddings Model.
type Config struct {
	// Size of the embedding vectors.
	Size int
	// Whether to return the `ZeroEmbedding` in case the key doesn't exist in
	// the embeddings store.
	// If it is false, nil is returned instead, so the caller has more
	// responsibility but also more control.
	UseZeroEmbedding bool
	// The name of the store.Store get from a store.Repository for the
	// data handled by the embeddings model.
	StoreName string
	// A trainable model allows its Embedding parameters to have gradient
	// values that can be propagated. When set to false, gradients handling is
	// disabled.
	Trainable bool
}

var _ nn.ParamsTraverser = &Model[string]{}

// A Model for handling embeddings.
type Model[K Key] struct {
	nn.Module
	Config
	// ZeroEmbedding is used as a fallback value for missing embeddings.
	//
	// If Config.UseZeroEmbedding is true, ZeroEmbedding is initialized
	// as a zero-vector of size Config.Size, otherwise it is set to nil.
	ZeroEmbedding nn.Param
	// Database where embeddings are stored.
	Store store.Store
	// embeddingsWithGrad is a private map filled with all those
	// embedding-parameters that have a gradient value attached.
	//
	// Parameters whose gradient is "zeroed" (see ag.Node.ZeroGrad) are
	// automatically removed from this map, thus freeing resources that would
	// otherwise be kept in memory indefinitely.
	//
	// In many simple use cases, as long as gradients are regularly zeroed,
	// the automatic mechanism described above will prevent the memory from
	// being cluttered with too many unused values.
	//
	// For other special or peculiar usages, for example if gradients are not
	// cleared regularly or at all, you might need to clear this map explicitly,
	// by calling the dedicated method ClearEmbeddingsWithGrad.
	embeddingsWithGrad map[string]nn.Param
	// This map is maintained in parallel with embeddingsWithGrad.
	// An Embedding parameter doesn't keep any internal value; everything is
	// rather delegated to the model, or the model's store.
	// The Embedding values stored in embeddingsWithGrad don't contain any
	// gradient value; instead the Model provides private methods allowing
	// reading and writing gradients by key, which are stored here.
	grads map[string]mat.Matrix
	mu    sync.RWMutex
}

func init() {
	gob.Register(&Model[string]{})
	gob.Register(&Model[[]byte]{})
}

// New returns a new embeddings Model.
//
// It panics in case of errors getting the Store from the Repository.
func New[T float.DType, K Key](conf Config, repo store.Repository) *Model[K] {
	st, err := repo.Store(conf.StoreName)
	if err != nil {
		panic(fmt.Errorf("embeddings: error getting Store %#v: %w", conf.StoreName, err))
	}

	var zeroEmb nn.Param = nil
	if conf.UseZeroEmbedding {
		zeroEmb = nn.NewParam(mat.NewEmptyVecDense[T](conf.Size)).WithGrad(false)
	}
	return &Model[K]{
		Config:        conf,
		ZeroEmbedding: zeroEmb,
		Store:         &store.PreventStoreMarshaling{Store: st},
	}
}

// TraverseParams allows embeddings with gradients to be traversed for optimization.
func (m *Model[K]) TraverseParams(callback func(param nn.Param)) {
	if m.ZeroEmbedding != nil {
		callback(m.ZeroEmbedding)
	}
	for _, emb := range m.embeddingsWithGrad {
		callback(emb)
	}
}

// Count counts how many embedding key/value pairs are currently stored.
// It panics in case of reading errors.
func (m *Model[_]) Count() int {
	n, err := m.Store.KeysCount()
	if err != nil {
		panic(fmt.Errorf("embeddings: error counting keys in store: %w", err))
	}
	return n
}

// CountEmbeddingsWithGrad counts how many embedding key/value pairs are currently active.
// It panics in case of reading errors.
func (m *Model[_]) CountEmbeddingsWithGrad() int {
	return len(m.embeddingsWithGrad)
}

// Embedding returns the Embedding parameter associated with the given key,
// also reporting whether the key was found in the store.
//
// Even if an embedding parameter is not found in the store, a usable value
// is still returned; it's sufficient to set some data on it (value, payload)
// to trigger its creation on the store.
//
// It panics in case of errors reading from the underlying store.
func (m *Model[K]) Embedding(key K) (*Embedding[K], bool) {
	if e, ok := m.embeddingsWithGrad[stringifyKey(key)]; ok {
		return e.(*Embedding[K]), true
	}

	exists, err := m.Store.Contains(encodeKey(key))
	if err != nil {
		panic(err)
	}
	e := &Embedding[K]{
		model: m,
		key:   key,
	}
	return e, exists
}

// EmbeddingFast is like Embedding, but skips the existence checks.
// Depending on the store implementation, this might save a considerable
// amount of time.
func (m *Model[K]) EmbeddingFast(key K) *Embedding[K] {
	if e, ok := m.embeddingsWithGrad[stringifyKey(key)]; ok {
		return e.(*Embedding[K])
	}
	return &Embedding[K]{
		model: m,
		key:   key,
	}
}

// Encode returns the embedding values associated with the input keys.
//
// The value are returned as Node(s) already inserted in the graph.
//
// Missing embedding values can be either nil or ZeroEmbedding, according
// to the Model's Config.
func (m *Model[K]) Encode(keys []K) []ag.Node {
	nodes := make([]ag.Node, len(keys))

	// reuse the same node for the same key
	cache := make(map[string]ag.Node, len(keys))

	for i, key := range keys {
		strKey := stringifyKey(key)

		if v, ok := cache[strKey]; ok {
			nodes[i] = v
			continue
		}

		var n ag.Node
		if e, ok := m.Embedding(key); ok {
			n = e
		} else {
			n = m.ZeroEmbedding
		}

		nodes[i] = n
		cache[strKey] = n
	}
	return nodes
}

// ClearEmbeddingsWithGrad empties the memory of visited embeddings with
// non-null gradient value.
func (m *Model[_]) ClearEmbeddingsWithGrad() {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.grads = nil
	m.embeddingsWithGrad = nil
}

// UseRepository allows the Model to use a Store from the given Repository.
//
// It only works if a store is not yet present. This can only happen in
// special situations, for example upon an Embedding model being deserialized,
// or when manually instantiating and handling a Model (i.e. bypassing New).
func (m *Model[_]) UseRepository(repo store.Repository) error {
	st, err := repo.Store(m.StoreName)
	if err != nil {
		return err
	}
	if m.storeExists() {
		return fmt.Errorf("a Store is already set on this embeddings.Model")
	}
	m.Store = &store.PreventStoreMarshaling{Store: st}
	return nil
}

func (m *Model[_]) storeExists() bool {
	switch s := m.Store.(type) {
	case nil:
		return false
	case store.PreventStoreMarshaling:
		return s.Store != nil
	case *store.PreventStoreMarshaling:
		return s.Store != nil
	default:
		return true
	}
}

func (m *Model[K]) getGrad(key K) (grad mat.Matrix, exists bool) {
	if !m.Trainable {
		return nil, false
	}

	m.mu.RLock()
	grad, exists = m.grads[stringifyKey(key)]
	m.mu.RUnlock()
	return
}

func (m *Model[K]) accGrad(e *Embedding[K], gx mat.Matrix) {
	if !m.Trainable {
		return
	}
	key := stringifyKey(e.key)

	m.mu.Lock()
	defer m.mu.Unlock()

	grad, exists := m.grads[key]
	if exists {
		grad.AddInPlace(gx)
		return
	}

	if m.grads == nil {
		m.grads = make(map[string]mat.Matrix)
		m.embeddingsWithGrad = make(map[string]nn.Param)
	}

	m.grads[key] = gx.Clone()
	m.embeddingsWithGrad[key] = e
}

func (m *Model[K]) zeroGrad(k K) {
	if !m.Trainable {
		return
	}
	key := stringifyKey(k)

	m.mu.Lock()
	defer m.mu.Unlock()

	grad, exists := m.grads[key]
	if !exists {
		return
	}

	mat.ReleaseMatrix(grad)
	delete(m.grads, key)
	delete(m.embeddingsWithGrad, key)
}
