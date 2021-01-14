// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat64

import (
	"encoding/binary"
	"encoding/gob"
	"math"
)

func init() {
	gob.Register(&Dense{})
}

// MarshalBinary marshals a Dense matrix into binary form.
func (d Dense) MarshalBinary() ([]byte, error) {
	data := make([]byte, 8+d.size*8)
	binary.LittleEndian.PutUint32(data, uint32(d.rows))
	binary.LittleEndian.PutUint32(data[4:], uint32(d.cols))
	for i, v := range d.data {
		binary.LittleEndian.PutUint64(data[8+i*8:], math.Float64bits(v))
	}
	return data, nil
}

// UnmarshalBinary unmarshals a binary representation of a Dense matrix.
func (d *Dense) UnmarshalBinary(data []byte) error {
	d.viewOf = nil
	d.fromPool = false
	d.rows = int(binary.LittleEndian.Uint32(data))
	d.cols = int(binary.LittleEndian.Uint32(data[4:]))
	d.size = d.rows * d.cols
	d.data = make([]Float, d.size)
	for i := range d.data {
		d.data[i] = math.Float64frombits(binary.LittleEndian.Uint64(data[8+i*8:]))
	}
	return nil
}
