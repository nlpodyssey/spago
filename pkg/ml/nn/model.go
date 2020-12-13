// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nn

import (
	"fmt"
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/utils"
	"io"
	"reflect"
	"strings"
)

// Model contains the serializable parameters.
type Model interface {
	// NewProc returns a new processor to execute the forward step.
	NewProc(ctx Context) Processor
}

// ForEachParam iterate all the parameters of a model also exploring the sub-parameters recursively.
// TODO: don't loop the field every time, use a lazy initialized "params list" instead
func ForEachParam(m Model, callback func(param *Param)) {
	forEachParam(m, callback, true)
}

// ForEachParamStrict iterate all the parameters of a model without exploring the sub-models.
func ForEachParamStrict(m Model, callback func(param *Param)) {
	forEachParam(m, callback, false)
}

// ForEachParam iterate all the parameters of a model also exploring the sub-parameters recursively.
// TODO: don't loop the field every time, use a lazy initialized "params list" instead
func forEachParam(m interface{}, callback func(param *Param), exploreSubModels bool) {
	utils.ForEachField(m, func(field interface{}, name string, tag reflect.StructTag) {
		switch item := field.(type) {
		case *Param:
			if item.name == "" {
				item.name = strings.ToLower(name)
			}
			item.pType = ToType(tag.Get("type"))
			callback(item)
		case Model:
			if exploreSubModels {
				forEachParam(item, callback, true)
			}
		case []*Param:
			for _, p := range item {
				if p.name == "" {
					p.name = strings.ToLower(name)
				}
				p.pType = ToType(tag.Get("type"))
				callback(p)
			}
		case []Model:
			if exploreSubModels {
				for _, m := range item {
					forEachParam(m, callback, true)
				}
			}
		default:
			v := reflect.ValueOf(item)
			switch v.Kind() {
			case reflect.Slice:
				length := v.Len()
				for i := 0; i < length; i++ {
					if m, ok := v.Index(i).Interface().(Model); ok {
						if exploreSubModels {
							forEachParam(m, callback, true)
						} else {
							return // skip
						}
					} else {
						switch v.Index(i).Kind() {
						case reflect.Struct, reflect.Ptr:
							if tag.Get("type") == "params" {
								forEachParam(item, callback, exploreSubModels)
							} else {
								return // skip
							}
						default:
							return // skip
						}
					}
				}
			case reflect.Map:
				mapRange := v.MapRange()
				for mapRange.Next() {
					key := ""
					switch k := mapRange.Key().Interface().(type) {
					case string:
						key = k
					case int:
						key = fmt.Sprintf("%d", k)
					default:
						return // skip map if the key is not a string or an int
					}
					// TODO: map of *Models
					p, ok := mapRange.Value().Interface().(*Param)
					if !ok {
						return // skip the map if the value is not a *Param
					}
					if p.name == "" {
						p.name = strings.ToLower(fmt.Sprintf("%s.%s", name, key))
					}
					p.pType = ToType(tag.Get("type"))
					callback(p)
				}
			case reflect.Struct, reflect.Ptr:
				if tag.Get("type") == "params" {
					forEachParam(item, callback, exploreSubModels)
				}
			}
		}
	})
}

type ParamsIterator interface {
	ParamsList() []*Param
}

var _ ParamsIterator = &DefaultParamsIterator{}

type DefaultParamsIterator struct {
	models []Model
}

func NewDefaultParamsIterator(models ...Model) *DefaultParamsIterator {
	return &DefaultParamsIterator{models: models}
}

func (i *DefaultParamsIterator) ParamsList() []*Param {
	params := make([]*Param, 0)
	for _, model := range i.models {
		ForEachParam(model, func(param *Param) {
			params = append(params, param)
		})
	}
	return params
}

// ZeroGrad set the gradients of all model's parameters (including sub-params) to zeros.
// TODO: use ParamsIterator?
func ZeroGrad(m Model) {
	ForEachParam(m, func(param *Param) {
		param.ZeroGrad()
	})
}

// ClearSupport clears the support structure of all model's parameters (including sub-params).
// TODO: use ParamsIterator?
func ClearSupport(m Model) {
	ForEachParam(m, func(param *Param) {
		param.ClearPayload()
	})
}

// TODO: use ParamsIterator?
func DumpParamsVector(model Model) *mat.Dense {
	data := make([]float64, 0)
	ForEachParam(model, func(param *Param) {
		data = append(data, param.Value().Data()...)
	})
	return mat.NewVecDense(data)
}

// TODO: use ParamsIterator?
func LoadParamsVector(model Model, vector *mat.Dense) {
	data := vector.Data()
	offset := 0
	ForEachParam(model, func(param *Param) {
		size := param.Value().Size()
		param.Value().SetData(data[offset : offset+size])
		offset += size
	})
}

// MakeNewModels return n new models.
// The callback is delegated to return a new model for each i-item.
func MakeNewModels(n int, callback func(i int) Model) []Model {
	lst := make([]Model, n)
	for i := 0; i < n; i++ {
		lst[i] = callback(i)
	}
	return lst
}

var _ utils.Serializer = &ParamsSerializer{}
var _ utils.Deserializer = &ParamsSerializer{}

type ParamsSerializer struct {
	Model
}

func NewParamsSerializer(m Model) *ParamsSerializer {
	return &ParamsSerializer{Model: m}
}

// Serialize dumps the params values to the writer.
// TODO: use ParamsIterator?
func (m *ParamsSerializer) Serialize(w io.Writer) (n int, err error) {
	ForEachParam(m, func(param *Param) {
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
// TODO: use ParamsIterator?
func (m *ParamsSerializer) Deserialize(r io.Reader) (n int, err error) {
	ForEachParam(m, func(param *Param) {
		cnt, err2 := mat.UnmarshalBinaryFrom(param.Value(), r)
		n += cnt
		if err2 != nil {
			err = err2
			return
		}
	})
	return n, err
}
