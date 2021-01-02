// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package kvdb

import (
	"github.com/dgraph-io/badger/v2"
	"log"
	"os"
)

// KeyValueDB is an interface implemented by key-value databases which spaGO
// can use.
type KeyValueDB interface {
	Put(key []byte, value []byte) error
	Get(key []byte) ([]byte, bool, error)
	Keys() ([]string, error)
	DropAll() error
	Close() error
}

// Config provides configuration parameters for KeyValueDB.
type Config struct {
	Path     string
	ReadOnly bool
	ForceNew bool
}

// NewDefaultKeyValueDB returns a new KeyValueDB.
func NewDefaultKeyValueDB(config Config) KeyValueDB {
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
	return &badgerBackend{
		Config: config,
		db:     db,
	}
}

var _ KeyValueDB = &badgerBackend{}

type badgerBackend struct {
	Config
	db *badger.DB
}

// Close closes the underlying DB.
// It's crucial to call it to ensure all the pending updates make their way to disk.
func (m *badgerBackend) Close() error {
	return m.db.Close()
}

// DropAll would drop all the data stored.
// Readings or writings performed during this operation may result in panics.
func (m *badgerBackend) DropAll() error {
	return m.db.DropAll()
}

func (m *badgerBackend) Keys() ([]string, error) {
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

func (m *badgerBackend) Put(key []byte, value []byte) error {
	return m.db.Update(func(txn *badger.Txn) error {
		entry := badger.NewEntry(key, value)
		err := txn.SetEntry(entry)
		return err // end view
	})
}

func (m *badgerBackend) Get(key []byte) (value []byte, ok bool, err error) {
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
