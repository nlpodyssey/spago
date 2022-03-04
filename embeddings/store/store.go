// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package store

import (
	"encoding/gob"
	"fmt"
)

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
	Put(key []byte, value any) error
	// Get attempts to fetch the value associated with the key, assigning it
	// to the given parameter, and returns a flag which reports whether
	// the key has been found or not.
	Get(key []byte, value any) (bool, error)
}

// PreventStoreMarshaling can wrap any Store implementation, embedding it to
// re-expose the full interface, overriding binary marshaling methods in order
// to produce (and expect) empty data.
//
// In the context of spaGO models, this wrapper is a convenient way to
// prevent Store public fields from being marshaled when models are
// serialized (for example, using gob).
type PreventStoreMarshaling struct {
	Store
}

// MarshalBinary satisfies encoding.BinaryMarshaler interface.
// It always produces empty data (nil) and no error.
func (PreventStoreMarshaling) MarshalBinary() ([]byte, error) {
	return nil, nil
}

// UnmarshalBinary satisfies encoding.BinaryUnmarshaler interface.
// It only accepts empty data (nil or zero-length slice), producing no
// side effects at all. If data is not blank, it returns an error.
func (PreventStoreMarshaling) UnmarshalBinary(data []byte) error {
	if len(data) != 0 {
		return fmt.Errorf("PreventStoreMarshaling.UnmarshalBinary: empty data expected, actual data len %d", len(data))
	}
	return nil
}

func init() {
	gob.Register(PreventStoreMarshaling{})
}
