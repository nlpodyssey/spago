// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package emb

import (
	"bufio"
	"bytes"
	"github.com/dgraph-io/badger"
	"github.com/gosuri/uiprogress"
	"io"
	"log"
	"os"
	"saientist.dev/spago/pkg/mat"
	"saientist.dev/spago/pkg/mat/f64utils"
	"saientist.dev/spago/pkg/ml/nn"
	"saientist.dev/spago/pkg/ml/optimizers/gd"
	"saientist.dev/spago/pkg/utils"
)

type Map struct {
	Config
	db *badger.DB
}

type Config struct {
	// size of each embedding vector
	Dim int
	// where to store the embedding map
	Path string
	// whether the embedding map is read only
	ReadOnly bool
}

// NewMapConfig returns the configuration based on the given arguments.
func NewMapConfig(dim int, path string, readOnly bool) Config {
	return Config{
		Dim:      dim,
		Path:     path,
		ReadOnly: readOnly,
	}
}

// NewMap returns a new empty embedding map.
func NewMap(config Config) *Map {
	db, err := badger.Open(
		badger.DefaultOptions(config.Path).
			WithReadOnly(config.ReadOnly).
			WithSyncWrites(false).
			WithLogger(nil),
	)
	if err != nil {
		log.Fatal(err)
	}
	return &Map{Config: config, db: db}
}

// Load inserts the pre-trained embeddings into the map.
func (m *Map) Load(filename string) {
	count, err := utils.CountLines(filename)
	if err != nil {
		log.Fatal(err)
	}

	uip := uiprogress.New()
	bar := uip.AddBar(count)
	bar.AppendCompleted().PrependElapsed()
	uip.Start() // start bar rendering
	defer uip.Stop()

	file, err := os.Open(filename)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	lineCount := 0
	for scanner.Scan() {
		lineCount++
		bar.Incr()
		if lineCount > 1 { // skip header
			line := scanner.Text()
			key := utils.BeforeSpace(line)
			strVec := utils.AfterSpace(line)
			data, err := f64utils.StrToFloat64Slice(strVec)
			if err != nil {
				log.Fatal(err)
			}
			m.SetVec(key, mat.NewVecDense(data))
		}
	}
	if err := scanner.Err(); err != nil {
		log.Fatal(err)
	}
}

// Close the embeddings map.
func (m *Map) Close() {
	if err := m.db.Close(); err != nil {
		log.Fatal(err)
	}
}

// Lookup
func (m *Map) Lookup(ks ...string) []*Embedding {
	result := make([]*Embedding, len(ks))
	cache := make(map[string]*Embedding)
	for i, k := range ks {
		if item, ok := cache[k]; ok {
			result[i] = item
		} else {
			var newEmbedding *Embedding
			if value, ok := m.getEntry(k); !ok {
				newEmbedding = nil
			} else {
				param := nn.NewParam(value.vec) // TODO: set requireGrad = !m.ReadOnly
				param.SetName(k)
				param.SetSupport(value.supp)
				newEmbedding = &Embedding{Param: param, storage: m}
			}
			cache[k] = newEmbedding
			result[i] = newEmbedding
		}
	}
	return result
}

// SetVec
func (m *Map) SetVec(key string, vector *mat.Dense) {
	var buf bytes.Buffer
	if _, err := mat.MarshalBinaryTo(vector, &buf); err != nil {
		log.Fatal(err)
	}
	if _, err := gd.MarshalBinaryTo(gd.NewEmptySupport(), &buf); err != nil {
		log.Fatal(err)
	}
	err := m.db.Update(func(txn *badger.Txn) error {
		e := badger.NewEntry([]byte(key), buf.Bytes())
		err := txn.SetEntry(e)
		return err // end view
	})
	if err != nil {
		log.Fatal(err)
	}
}

// Keys
func (m *Map) Keys() ([]string, error) {
	var lst []string
	err := m.db.View(func(txn *badger.Txn) error {
		opts := badger.DefaultIteratorOptions
		opts.PrefetchValues = false
		it := txn.NewIterator(opts)
		defer it.Close()
		for it.Rewind(); it.Valid(); it.Next() {
			item := it.Item()
			k := item.Key()
			lst = append(lst, string(k))
		}
		return nil // end view
	})
	return lst, err
}

// the value of the embeddings map
type entry struct {
	// the dense vector of the embedding
	vec *mat.Dense
	// the support structure used by the gradient descent optimizer (Adam, AdaGrad, ...)
	supp *gd.Support
}

func marshalBinaryTo(v *entry, w io.Writer) (int, error) {
	n, err := mat.MarshalBinaryTo(v.vec, w)
	if err != nil {
		return n, err
	}
	n2, err := gd.MarshalBinaryTo(v.supp, w)
	n += n2
	if err != nil {
		return n, err
	}
	return n, err
}

func unmarshalBinaryFrom(r io.Reader) (*entry, int, error) {
	vec, n, err := mat.NewUnmarshalBinaryFrom(r)
	if err != nil {
		return nil, n, err
	}
	supp, n2, err := gd.NewUnmarshalBinaryFrom(r)
	n += n2
	if err != nil {
		return nil, n, err
	}
	return &entry{vec: vec, supp: supp}, n, err
}

func (m *Map) getEntry(key string) (*entry, bool) {
	var value *entry
	err := m.db.View(func(txn *badger.Txn) error {
		item, err := txn.Get([]byte(key))
		if err != nil {
			return err
		}
		valCopy, err := copyValue(item)
		if err != nil {
			return err
		}
		value, _, err = unmarshalBinaryFrom(bytes.NewReader(valCopy))
		return nil // end view
	})
	if err != nil {
		if err == badger.ErrKeyNotFound {
			return nil, false
		}
		log.Fatal(err)
	}
	return value, true
}

func copyValue(item *badger.Item) ([]byte, error) {
	var valCopy []byte
	err := item.Value(func(val []byte) error {
		valCopy = append([]byte{}, val...)
		return nil
	})
	return valCopy, err
}

func (m *Map) update(e *Embedding) (int, error) {
	var buf bytes.Buffer
	n, err := marshalBinaryTo(&entry{
		vec:  e.Value().(*mat.Dense),
		supp: e.Support(),
	}, &buf)
	err = m.db.Update(func(txn *badger.Txn) error {
		entry := badger.NewEntry([]byte(e.Param.Name()), buf.Bytes())
		err := txn.SetEntry(entry)
		return err // end view
	})
	return n, err
}
