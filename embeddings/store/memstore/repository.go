// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package memstore

import (
	"sync"

	"github.com/nlpodyssey/spago/embeddings/store"
)

// A Repository of in-memory key-value stores.
type Repository struct {
	stores map[string]*Store // []byte keys are stringified here
	mu     sync.Mutex
}

// NewRepository returns a new empty Repository of in-memory key-value Stores.
func NewRepository() *Repository {
	return &Repository{
		stores: make(map[string]*Store, 0),
	}
}

// Store returns an in-memory data store by name.
func (r *Repository) Store(name string) (store.Store, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if n, ok := r.stores[name]; ok {
		return n, nil
	}

	s := newStore(name)
	r.stores[name] = s
	return s, nil
}

// DropAll removes all data and drops all stores.
func (r *Repository) DropAll() error {
	r.mu.Lock()
	defer r.mu.Unlock()

	for name := range r.stores {
		delete(r.stores, name)
	}
	return nil
}
