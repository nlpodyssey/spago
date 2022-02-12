// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package kvdb

import (
	"github.com/dgraph-io/badger/v3"
	"log"
	"os"
)

// KeyValueDB is a key-value database which spaGO can use to efficiently store
// large data.
type KeyValueDB struct {
	Config
	db *badger.DB
}

// Config provides configuration parameters for KeyValueDB.
type Config struct {
	Path     string
	ReadOnly bool
	ForceNew bool
}

// NewDefaultKeyValueDB returns a new KeyValueDB.
func NewDefaultKeyValueDB(config Config) *KeyValueDB {
	if config.ForceNew {
		err := os.RemoveAll(config.Path)
		if err != nil {
			log.Println(err)
		}
	}
	options := badger.DefaultOptions(config.Path).
		WithReadOnly(config.ReadOnly).
		WithSyncWrites(false).
		WithLogger(nil)

	db, err := badger.Open(options)
	if err != nil {
		log.Fatal(err)
	}
	return &KeyValueDB{
		Config: config,
		db:     db,
	}
}

// MarshalBinary prevents KeyValueDB to be encoded to binary representation.
//
// It never makes sense to encode/decode a KeyValueDB value, for example
// when used as a model parameter. So this method always returns nil, and
// no errors.
func (KeyValueDB) MarshalBinary() ([]byte, error) {
	return nil, nil
}

// UnmarshalBinary prevents KeyValueDB to be decoded from binary representation.
//
// It never makes sense to encode/decode a KeyValueDB value, for example
// when used as a model parameter. So this method always does not modify the
// receiver in any way, and never returns errors.
func (*KeyValueDB) UnmarshalBinary([]byte) error {
	return nil
}

// Close closes the underlying DB.
// It's crucial to call it to ensure all the pending updates make their way to disk.
func (m *KeyValueDB) Close() error {
	return m.db.Close()
}

// DropAll would drop all the data stored.
// Readings or writings performed during this operation may result in panics.
func (m *KeyValueDB) DropAll() error {
	return m.db.DropAll()
}

// Keys returns all the keys from the DB.
func (m *KeyValueDB) Keys() ([]string, error) {
	var keys []string
	err := m.db.View(func(txn *badger.Txn) error {
		opts := badger.DefaultIteratorOptions
		opts.PrefetchValues = false
		it := txn.NewIterator(opts)
		defer it.Close()
		for it.Rewind(); it.Valid(); it.Next() {
			keys = append(keys, string(it.Item().Key()))
		}
		return nil // end view
	})
	return keys, err
}

// Put sets a new key/value pair in the DB.
func (m *KeyValueDB) Put(key []byte, value []byte) error {
	return m.db.Update(func(txn *badger.Txn) error {
		entry := badger.NewEntry(key, value)
		err := txn.SetEntry(entry)
		return err // end view
	})
}

// Get returns the value associated to the given key, if it exists.
func (m *KeyValueDB) Get(key []byte) (value []byte, ok bool, err error) {
	err = m.db.View(func(txn *badger.Txn) error {
		item, err := txn.Get(key)
		if err != nil {
			return err
		}
		value, err = copyValue(item)
		if err != nil {
			return err
		}
		return nil // end view
	})
	switch {
	case err == nil:
		return value, true, nil
	case err == badger.ErrKeyNotFound:
		return nil, false, nil
	default:
		return nil, false, err
	}
}

func copyValue(item *badger.Item) ([]byte, error) {
	var valCopy []byte
	err := item.Value(func(val []byte) error {
		valCopy = append([]byte{}, val...)
		return nil
	})
	return valCopy, err
}
