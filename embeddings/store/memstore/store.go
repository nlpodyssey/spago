// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package memstore

import (
	"reflect"
	"sync"
)

// A Store of in-memory key-value pairs.
type Store struct {
	name string
	m    map[string]interface{}
	mu   sync.RWMutex
}

func newStore(name string) *Store {
	return &Store{
		name: name,
		m:    make(map[string]interface{}, 0),
	}
}

// The Name of the store.
func (s *Store) Name() string {
	return s.name
}

// DropAll drops all data from the store.
func (s *Store) DropAll() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	for k := range s.m {
		delete(s.m, k)
	}
	return nil
}

// Keys returns all the keys from the store.
func (s *Store) Keys() ([][]byte, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if len(s.m) == 0 {
		return nil, nil
	}
	keys := make([][]byte, 0, len(s.m))
	for k := range s.m {
		keys = append(keys, []byte(k))
	}
	return keys, nil
}

// KeysCount reports how many key/value pairs are in the store.
func (s *Store) KeysCount() (int, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	return len(s.m), nil
}

// Contains reports whether the given key is found in the store.
func (s *Store) Contains(key []byte) (bool, error) {
	s.mu.RLock()
	_, exists := s.m[string(key)]
	s.mu.RUnlock()

	return exists, nil
}

// Put sets a key/value pair in the store.
// If a value for the same key already exists in the store, it is
// overwritten with the new value.
func (s *Store) Put(key []byte, value interface{}) error {
	s.mu.Lock()
	s.m[string(key)] = value
	s.mu.Unlock()
	return nil
}

// Get attempts to fetch the value associated with the key, assigning it
// to the given parameter, and returns a flag which reports whether
// the key has been found or not.
func (s *Store) Get(key []byte, value interface{}) (bool, error) {
	s.mu.RLock()
	i, exists := s.m[string(key)]
	s.mu.RUnlock()

	if !exists {
		return false, nil
	}

	v := reflect.ValueOf(i)
	if v.Kind() == reflect.Pointer {
		v = v.Elem()
	}

	reflect.ValueOf(value).Elem().Set(v)

	return true, nil
}
