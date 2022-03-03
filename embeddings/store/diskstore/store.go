// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package diskstore

import (
	"encoding"
	"encoding/base64"
	"encoding/gob"
	"fmt"
	"path/filepath"

	"github.com/dgraph-io/badger/v3"
)

// A Store is an on-disk key-value database.
type Store struct {
	name string
	db   *badger.DB
}

// Prefix prepended to all store filenames to circumvent the problem of
// empty-string names ("").
const storeFileNamePrefix = "store_"

var nameEncoding = base64.URLEncoding.WithPadding(base64.NoPadding)

func newStore(path, name string, readOnly bool) (*Store, error) {
	// The name is encoded using "URL and Filename safe" Base 64 encoding.
	// However, an empty name ("") would still be a problematic value,
	// so we always prepend a fixed prefix value.
	encodedName := storeFileNamePrefix + nameEncoding.EncodeToString([]byte(name))

	dbPath := filepath.Join(path, encodedName)

	opt := badger.DefaultOptions(dbPath).WithLoggingLevel(badger.WARNING)
	if readOnly {
		opt = opt.WithReadOnly(true)
	}

	db, err := badger.Open(opt)
	if err != nil {
		return nil, err
	}
	s := &Store{
		name: name,
		db:   db,
	}
	return s, nil
}

func (s *Store) close() error {
	if err := s.db.Close(); err != nil {
		return err
	}
	s.db = nil
	return nil
}

// The Name of the store.
func (s *Store) Name() string {
	return s.name
}

// DropAll drops all data from the store.
//
// It panics in read-only mode.
//
// New readings performed during this operation may result in panics.
func (s *Store) DropAll() error {
	return s.db.DropAll()
}

// Keys returns all the keys from the store.
func (s *Store) Keys() ([][]byte, error) {
	var keys [][]byte

	opts := badger.DefaultIteratorOptions
	opts.PrefetchValues = false // key-only iteration

	err := s.db.View(func(txn *badger.Txn) error {
		it := txn.NewIterator(opts)
		defer it.Close()
		for it.Rewind(); it.Valid(); it.Next() {
			keys = append(keys, it.Item().KeyCopy(nil))
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	return keys, nil
}

// KeysCount reports how many key/value pairs are in the store.
func (s *Store) KeysCount() (int, error) {
	count := 0

	opts := badger.DefaultIteratorOptions
	opts.PrefetchValues = false // key-only iteration

	err := s.db.View(func(txn *badger.Txn) error {
		it := txn.NewIterator(opts)
		defer it.Close()
		for it.Rewind(); it.Valid(); it.Next() {
			count++
		}
		return nil
	})
	if err != nil {
		return 0, err
	}
	return count, nil
}

// Contains reports whether the given key is found in the store.
func (s *Store) Contains(key []byte) (bool, error) {
	err := s.db.View(func(txn *badger.Txn) error {
		_, err := txn.Get(key)
		return err
	})

	if err == badger.ErrKeyNotFound {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	return true, nil
}

// Put sets a key/value pair in the store.
// If a value for the same key already exists in the store, it is
// overwritten with the new value.
func (s *Store) Put(key []byte, value any) (err error) {
	var bytesValue []byte
	switch vt := value.(type) {
	case []byte:
		bytesValue = vt
	case encoding.BinaryMarshaler:
		bytesValue, err = vt.MarshalBinary()
		if err != nil {
			return err
		}
	case gob.GobEncoder:
		bytesValue, err = vt.GobEncode()
		if err != nil {
			return err
		}
	default:
		return fmt.Errorf("unsupported value type %T", value)
	}

	entry := badger.NewEntry(key, bytesValue)

	return s.db.Update(func(txn *badger.Txn) error {
		return txn.SetEntry(entry)
	})
}

// Get attempts to fetch the value associated with the key, assigning it
// to the given parameter, and returns a flag which reports whether
// the key has been found or not.
func (s *Store) Get(key []byte, value any) (bool, error) {
	var bytesValue []byte

	err := s.db.View(func(txn *badger.Txn) error {
		item, err := txn.Get(key)
		if err != nil {
			return err
		}
		bytesValue, err = item.ValueCopy(nil)
		return err
	})

	if err == badger.ErrKeyNotFound {
		return false, nil
	}
	if err != nil {
		return false, err
	}

	switch vt := value.(type) {
	case *[]byte:
		*vt = bytesValue
	case encoding.BinaryUnmarshaler:
		if err = vt.UnmarshalBinary(bytesValue); err != nil {
			return true, err
		}
	case gob.GobDecoder:
		if err = vt.GobDecode(bytesValue); err != nil {
			return true, err
		}
	default:
		return true, fmt.Errorf("unsupported value type %T", value)
	}

	return true, nil
}
