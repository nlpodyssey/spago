// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package store

// A Repository is a logical grouping of different stores.
type Repository interface {
	// Store returns a data store by name.
	Store(name string) (Store, error)
	// DropAll removes all data and drops all stores.
	DropAll() error
}

// Store is an interface to a single key-value store.
type Store interface {
	// The Name of the store.
	Name() string
	// DropAll drops all data from the store.
	DropAll() error
	// Keys returns all the keys from the store.
	Keys() ([][]byte, error)
	// KeysCount reports how many key/value pairs are in the store.
	KeysCount() (int, error)
	// Contains reports whether the given key is found in the store.
	Contains(key []byte) (bool, error)
	// Put sets a key/value pair in the store.
	// If a value for the same key already exists in the store, it is
	// overwritten with the new value.
	Put(key []byte, value interface{}) error
	// Get attempts to fetch the value associated with the key, assigning it
	// to the given parameter, and returns a flag which reports whether
	// the key has been found or not.
	Get(key []byte, value interface{}) (bool, error)
}
