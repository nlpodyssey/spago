// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package kvdb

import (
	"github.com/dgraph-io/badger/v2"
	"log"
	"os"
)

type KeyValueDB interface {
	Put(key []byte, value []byte) error
	Get(key []byte) ([]byte, bool, error)
	Keys() ([]string, error)
	Close() error
}

type Config struct {
	Path     string
	ReadOnly bool
	ForceNew bool
}

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

func (m *badgerBackend) Close() error {
	return m.db.Close()
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
