// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nn

import (
	"fmt"
	"github.com/nlpodyssey/spago/pkg/utils"
	"reflect"
	"strings"
)

// paramsTraversal allows the traversal of Model parameters.
// The given callback is invoked for each parameter of the Model.
// If exploreSubModels is true, every nested Model and its parameters are
// also visited.
type paramsTraversal struct {
	callback         func(param *Param)
	exploreSubModels bool
}

// newParamsTraversal returns a new paramsTraversal.
func newParamsTraversal(callback func(param *Param), exploreSubModels bool) paramsTraversal {
	return paramsTraversal{
		callback:         callback,
		exploreSubModels: exploreSubModels,
	}
}

// walk iterates through all the parameters of m.
// TODO: don't loop the field every time, use a lazy initialized "params list" instead
func (pt paramsTraversal) walk(m interface{}) {
	utils.ForEachField(m, func(field interface{}, name string, tag reflect.StructTag) {
		switch item := field.(type) {
		case *Param:
			pt.walkParam(item, name, tag)
		case Model:
			pt.walkModel(item)
		case []*Param:
			pt.walkParamSlice(item, name, tag)
		case []Model:
			pt.walkModelSlice(item)
		default:
			v := reflect.ValueOf(item)
			switch v.Kind() {
			case reflect.Slice:
				pt.walkGenericSlice(v, tag, item)
			case reflect.Map:
				pt.walkGenericMap(v, name, tag)
			case reflect.Struct, reflect.Ptr:
				pt.walkGenericStructOrPtr(tag, item)
			}
		}
	})
}

func (pt paramsTraversal) walkParam(item *Param, name string, tag reflect.StructTag) {
	if item.name == "" {
		item.name = strings.ToLower(name)
	}
	item.pType = ToType(tag.Get("type"))
	pt.callback(item)
}

func (pt paramsTraversal) walkModel(item Model) {
	if pt.exploreSubModels {
		pt.walk(item)
	}
}

func (pt paramsTraversal) walkParamSlice(item []*Param, name string, tag reflect.StructTag) {
	for _, p := range item {
		if p.name == "" {
			p.name = strings.ToLower(name)
		}
		p.pType = ToType(tag.Get("type"))
		pt.callback(p)
	}
}

func (pt paramsTraversal) walkModelSlice(item []Model) {
	if pt.exploreSubModels {
		for _, m := range item {
			pt.walk(m)
		}
	}
}

func (pt paramsTraversal) walkGenericSlice(v reflect.Value, tag reflect.StructTag, item interface{}) {
	length := v.Len()
	for i := 0; i < length; i++ {
		if m, ok := v.Index(i).Interface().(Model); ok {
			if pt.exploreSubModels {
				pt.walk(m)
			} else {
				return // skip
			}
		} else {
			p := v.Index(i)
			switch p.Kind() {
			case reflect.Struct, reflect.Ptr:
				if tag.Get("type") == "params" {
					pt.walk(p.Interface())
				} else {
					return // skip
				}
			default:
				return // skip
			}
		}
	}
}

func (pt paramsTraversal) walkGenericMap(v reflect.Value, name string, tag reflect.StructTag) {
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
		pt.callback(p)
	}
}

func (pt paramsTraversal) walkGenericStructOrPtr(tag reflect.StructTag, item interface{}) {
	if tag.Get("type") == "params" {
		pt.walk(item)
	}
}
