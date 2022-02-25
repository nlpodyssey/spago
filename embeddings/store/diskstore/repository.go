// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package diskstore

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"

	"github.com/nlpodyssey/spago/embeddings/store"
)

// A Repository of key-value Stores persisted on disk.
type Repository struct {
	path     string
	stores   map[string]*Store // []byte keys are stringified here
	readOnly bool
	mu       sync.Mutex
}

// Mode identifies the strategy adopted by Repositories and Stores in respect
// to the data from disk.
type Mode uint8

const (
	// ReadWriteMode allows full reading and writing access to data stored on
	// disk. If no data exists yet, a new blank store is created.
	ReadWriteMode Mode = iota
	// ReadOnlyMode provides read-only access.
	// If no data exists yet on disk, an error is reported.
	ReadOnlyMode
)

// NewRepository returns a Repository bound to data located on disk at the
// given path, handled according to the specified mode.
//
// In ReadWriteMode, if the directory named path does not exist, it is
// created along with any necessary parents, setting its permissions bits
// to 0755 (rwxr-xr-x).
//
// Once you are done with its usage, it's important to call Repository.Close
// to ensure that any pending update on the repository's Stores is persisted
// to disk.
func NewRepository(path string, mode Mode) (*Repository, error) {
	r := &Repository{
		path:   path,
		stores: make(map[string]*Store, 0),
	}

	_, err := os.Stat(path)
	pathExists := err != nil

	switch mode {
	case ReadWriteMode:
		r.readOnly = false
		if pathExists {
			if err := os.MkdirAll(path, 0755); err != nil {
				return nil, err
			}
		}
	case ReadOnlyMode:
		r.readOnly = true
		if pathExists {
			return nil, fmt.Errorf("repository directory does not exist: %#v", path)
		}
	default:
		return nil, errors.New("invalid RepositoryMode")
	}

	return r, nil
}

// Store returns an on-disk data store by name.
//
// In ReadWriteMode, getting a store for the first time will cause its creation.
// In ReadOnlyMode, you can only get existing stores, otherwise an error is
// returned.
//
// Getting a store will cause the creation of a binding with the underlying
// data on disk. Getting the same store more than once will return the same
// underlying object and data binding (the operation is idempotent).
//
// Once you are done using all stores from a Repository, remember to call
// Repository.Close to ensure all data is correctly written on disk and
// resources are freed.
func (r *Repository) Store(name string) (store.Store, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if s, ok := r.stores[name]; ok {
		return s, nil
	}

	s, err := newStore(r.path, name, r.readOnly)
	if err != nil {
		return nil, err
	}
	r.stores[name] = s
	return s, nil
}

// DropAll removes all data and drops all stores.
//
// Before proceeding with data removal, any existing binding with files on disk
// is first closed. To avoid errors or unexpected behaviors, you must be
// sure that no Store object obtained before performing this operation will be
// used thereafter.
//
// This operation internally loops through each previously open store.
// In case of failures, the first error encountered is returned,
// aborting the operation; if this happens, not all stores might have
// been dropped properly.
func (r *Repository) DropAll() (err error) {
	if r.readOnly {
		return errors.New("the repository is read-only")
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	for name, s := range r.stores {
		if err := s.close(); err != nil {
			return fmt.Errorf("error closing store %#v before dropping: %w", name, err)
		}
		delete(r.stores, name)
	}

	// We might be tempted to just delete the whole directory r.path,
	// but it's better to keep it intact and delete its content instead.
	// In this way, the directory's permissions are kept, which is a nice
	// courtesy for the users in case they want to reuse the same path later.

	f, err := os.Open(r.path)
	if err != nil {
		return err
	}
	defer func() {
		if e := f.Close(); e != nil && err == nil {
			err = e
		}
	}()

	infos, err := f.Readdir(0)
	if err != nil {
		return err
	}

	for _, info := range infos {
		path := filepath.Join(r.path, info.Name())
		if err = os.RemoveAll(path); err != nil {
			return err
		}
	}

	return nil
}

// Close flushes pending updates of all previously used stores (if any), and
// closes any binding with data on disk, also freeing internal resources.
//
// In case of failures, the first error encountered is returned, aborting the
// closing operation; if this happens, not all stores might have been closed
// and flushed properly.
//
// This must be the last operation performed after using a Repository and its
// stores. You must be sure that no Store object obtained before performing
// this operation will be used thereafter.
func (r *Repository) Close() error {
	r.mu.Lock()
	defer r.mu.Unlock()

	for name, s := range r.stores {
		if err := s.close(); err != nil {
			return fmt.Errorf("error closing store %#v: %w", name, err)
		}
		delete(r.stores, name)
	}
	return nil
}
