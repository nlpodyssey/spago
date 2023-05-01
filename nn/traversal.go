// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nn

import (
	"fmt"
	"reflect"
	"sync"
)

// ParamsTraverser allows you to define a custom procedure to traverse the parameters of a model.
// If a model implements this procedure, it will take precedence over the regular parameters visit.
type ParamsTraverser interface {
	// TraverseParams visit each Param.
	TraverseParams(callback func(param *Param))
}

// paramsTraversal allows the traversal of Model parameters.
// The given paramsFunc is invoked for each parameter of the Model.
// If exploreSubModels is true, every nested Model and its parameters are
// also visited.
type paramsTraversal struct {
	paramsFunc       func(param *Param)
	modelsFunc       func(model Model)
	exploreSubModels bool
}

// walk iterates through all the parameters of m.
func (pt paramsTraversal) walk(m any) {
	if m, ok := m.(ParamsTraverser); ok {
		m.TraverseParams(pt.paramsFunc)
		return
	}
	forEachField(m, func(field any, name string) {
		v := reflect.ValueOf(field)
		switch v.Kind() {
		case reflect.Struct, reflect.Ptr, reflect.Interface:
			pt.walkStructOrPtr(field, name)
		case reflect.Slice, reflect.Array:
			pt.walkSlice(v, name)
		case reflect.Map:
			pt.walkMap(v, name)
		}
	})
}

func (pt paramsTraversal) walkStructOrPtr(item any, name string) bool {
	v := reflect.ValueOf(item)
	if v.Kind() == reflect.Ptr && v.Elem().Kind() != reflect.Struct {
		return false
	}
	switch itemT := item.(type) {
	case Module, *Module:
		// skip
	case *Param:
		if pt.paramsFunc != nil {
			pt.paramsFunc(itemT)
		}
	case ParamsTraverser:
		if pt.paramsFunc != nil {
			itemT.TraverseParams(pt.paramsFunc)
		}
		if m, ok := item.(Model); ok && pt.modelsFunc != nil {
			pt.modelsFunc(m)
		}
	case Model:
		if pt.exploreSubModels {
			if pt.modelsFunc != nil {
				pt.modelsFunc(itemT)
			}
			pt.walk(item)
		}
	case *sync.Map:
		pt.walkSyncMap(itemT, name)
	default:
		return false
	}
	return true
}

func (pt paramsTraversal) walkSyncMap(i *sync.Map, name string) {
	i.Range(func(key, value any) bool {
		switch k := key.(type) {
		case string:
			key = k
		case int:
			key = fmt.Sprintf("%d", k)
		default:
			return false // skip map if the key is not a string or an int
		}

		name := fmt.Sprintf("%s.%s", name, key)
		switch reflect.ValueOf(value).Kind() {
		case reflect.Struct, reflect.Ptr, reflect.Interface:
			if !pt.walkStructOrPtr(value, name) {
				return false
			}
		default:
			return false // skip
		}
		return true
	})
}

func (pt paramsTraversal) walkSlice(v reflect.Value, name string) {
	length := v.Len()
	for i := 0; i < length; i++ {
		p := v.Index(i)
		switch p.Kind() {
		case reflect.Struct, reflect.Ptr, reflect.Interface:
			if !pt.walkStructOrPtr(p.Interface(), name) {
				return
			}
		default:
			return // skip
		}
	}
}

func (pt paramsTraversal) walkMap(v reflect.Value, name string) {
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

		name := fmt.Sprintf("%s.%s", name, key)
		switch mapRange.Value().Kind() {
		case reflect.Struct, reflect.Ptr, reflect.Interface:
			if !pt.walkStructOrPtr(mapRange.Value().Interface(), name) {
				return
			}
		default:
			return // skip
		}
	}
}

// forEachField calls the paramsFunc for each field of the struct i.
func forEachField(i any, callback func(field any, name string)) {
	v := reflect.ValueOf(i)
	t := reflect.TypeOf(i)

	if v.Kind() == reflect.Ptr {
		v = v.Elem()
		t = t.Elem()
	}

	length := v.NumField()
	for i := 0; i < length; i++ {
		vField := v.Field(i)
		tField := t.Field(i)
		if vField.CanInterface() {
			if vField.Kind() == reflect.Ptr && vField.IsNil() {
				continue
			}
			callback(vField.Interface(), tField.Name)
		}
	}
}
