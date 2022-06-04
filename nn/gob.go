// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nn

import (
	"bufio"
	"encoding/gob"
	"io"
	"os"
)

// Dump saves a serialized object to a stream. This function uses Gob utility for serialization.
// Models, matrices, and all kinds of Gob serializable objects can be saved using this function.
func Dump(obj any, w io.Writer) error {
	bw := bufio.NewWriter(w)
	if err := gob.NewEncoder(bw).Encode(obj); err != nil {
		return err
	}
	err := bw.Flush()
	if err != nil {
		return err
	}
	return nil
}

// DumpToFile saves a serialized object to a file.
// See Dump for further details.
func DumpToFile[T any](obj T, filename string) error {
	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer func() {
		if e := f.Close(); e != nil && err == nil {
			err = e
		}
	}()
	if err = Dump(obj, f); err != nil {
		return err
	}
	return nil
}

// Load uses Gob to deserialize objects to memory.
func Load[T any](r io.Reader, args ...any) (T, error) {
	if len(args) == 0 {
		args = []any{nil}
	}
	var obj T
	if err := gob.NewDecoder(bufio.NewReader(r)).Decode(&obj); err != nil {
		return obj, err
	}
	if v, ok := any(obj).(interface{ AfterLoad(any) error }); ok {
		if err := v.AfterLoad(args[0]); err != nil {
			return obj, err
		}
	}
	return obj, nil
}

// LoadFromFile uses Gob to deserialize objects files to memory.
// See Load for further details.
func LoadFromFile[T any](filename string, args ...any) (T, error) {
	f, err := os.Open(filename)
	if err != nil {
		var obj T
		return obj, err
	}
	defer func() {
		if e := f.Close(); e != nil && err == nil {
			err = e
		}
	}()
	if len(args) > 0 {
		return Load[T](f, args[0])
	}
	return Load[T](f)
}
