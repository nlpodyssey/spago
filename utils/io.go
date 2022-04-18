// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package utils

import (
	"bufio"
	"encoding/gob"
	"os"
)

// SerializeToFile serializes obj to file, using gob encoding.
func SerializeToFile(filename string, obj any) (err error) {
	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer func() {
		if e := f.Close(); e != nil && err == nil {
			err = e
		}
	}()
	buf := bufio.NewWriter(f) // Buffered writing is essential to avoid memory leaks with large data
	err = gob.NewEncoder(buf).Encode(obj)
	if err != nil {
		return err
	}
	err = buf.Flush()
	if err != nil {
		return err
	}
	return nil
}

// DeserializeFromFile deserializes obj from file, using gob decoding.
func DeserializeFromFile(filename string, obj any) (err error) {
	f, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer func() {
		if e := f.Close(); e != nil && err == nil {
			err = e
		}
	}()
	err = gob.NewDecoder(bufio.NewReader(f)).Decode(obj)
	if err != nil {
		return err
	}
	return
}
