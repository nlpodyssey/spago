// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nn

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/utils"
	"io"
)

var _ utils.Serializer = &ParamsSerializer{}
var _ utils.Deserializer = &ParamsSerializer{}

// ParamsSerializer allows serialization and deserialization of all
// parameters of a given Model.
type ParamsSerializer struct {
	Model
}

// NewParamsSerializer returns a new ParamsSerializer.
func NewParamsSerializer(m Model) *ParamsSerializer {
	return &ParamsSerializer{
		Model: m,
	}
}

// Serialize dumps the params values to the writer.
func (m *ParamsSerializer) Serialize(w io.Writer) (n int, err error) {
	ForEachParam(m, func(param Param) {
		cnt, err2 := mat.MarshalBinaryTo(param.Value(), w)
		n += cnt
		if err2 != nil {
			err = err2
			return
		}
	})
	return n, err
}

// Deserialize assigns the params with the values obtained from the reader.
func (m *ParamsSerializer) Deserialize(r io.Reader) (n int, err error) {
	ForEachParam(m, func(param Param) {
		cnt, err2 := mat.UnmarshalBinaryFrom(param.Value(), r)
		n += cnt
		if err2 != nil {
			err = err2
			return
		}
	})
	return n, err
}
