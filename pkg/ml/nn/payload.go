// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nn

import (
	"bytes"
	"encoding/binary"
	mat "github.com/nlpodyssey/spago/pkg/mat32"
)

// Payload contains the support data used for example by the optimization methods
type Payload struct {
	Label int
	Data  []mat.Matrix
}

// NewPayload returns an empty support structure, not connected to any optimization method.
func NewPayload() *Payload {
	return &Payload{
		Label: 0, // important set the label to zero
		Data:  make([]mat.Matrix, 0),
	}
}

// MarshalBinary encodes the Payload into binary form.
func (p Payload) MarshalBinary() ([]byte, error) {
	buf := new(bytes.Buffer)

	binLabel := make([]byte, 8)
	binary.LittleEndian.PutUint64(binLabel, uint64(p.Label))
	buf.Write(binLabel)

	binLen := make([]byte, 4)
	binary.LittleEndian.PutUint32(binLen, uint32(len(p.Data)))
	buf.Write(binLen)

	for _, m := range p.Data {
		err := mat.MarshalBinaryMatrix(m, buf)
		if err != nil {
			return nil, err
		}
	}

	return buf.Bytes(), nil
}

// UnmarshalBinary decodes a Payload from binary form.
func (p *Payload) UnmarshalBinary(data []byte) error {
	p.Label = int(binary.LittleEndian.Uint64(data))
	dataLen := int(binary.LittleEndian.Uint32(data[8:]))

	var err error
	r := bytes.NewReader(data[12:])

	p.Data = make([]mat.Matrix, dataLen)
	for i := range p.Data {
		p.Data[i], err = mat.UnmarshalBinaryMatrix(r)
		if err != nil {
			return err
		}
	}
	return nil
}
