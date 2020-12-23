// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nn

import (
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"reflect"
)

type modelContextualizer struct {
	graph *ag.Graph
}

// newModelContextualizer returns a new modelContextualizer.
func newModelContextualizer(graph *ag.Graph) modelContextualizer {
	return modelContextualizer{
		graph: graph,
	}
}

func (mc modelContextualizer) contextualize(m Model) Model {
	return mc.contextualizeStruct(m).(Model)
}

func (mc modelContextualizer) contextualizeStruct(rawSource interface{}) interface{} {
	source := reflect.ValueOf(rawSource)
	sourceType := reflect.TypeOf(rawSource)

	sourceIsPointer := source.Kind() == reflect.Ptr
	if sourceIsPointer {
		sourceIsPointer = true
		source = source.Elem()
		sourceType = sourceType.Elem()
	}

	destPointer := reflect.New(sourceType)
	dest := destPointer.Elem()

	numFields := source.NumField()
	for fieldIndex := 0; fieldIndex < numFields; fieldIndex++ {
		sourceField, destField := source.Field(fieldIndex), dest.Field(fieldIndex)
		tag := sourceType.Field(fieldIndex).Tag

		if tag.Get("scope") == "processor" {
			continue // skip any initialization
		}

		switch sourceFieldT := sourceField.Interface().(type) {
		case Param:
			destField.Set(reflect.ValueOf(mc.contextualizeParam(sourceFieldT)))
		case []Param:
			destField.Set(reflect.ValueOf(mc.contextualizeParamSlice(sourceFieldT)))
		default:
			switch sourceField.Kind() {
			case reflect.Slice:
				destField.Set(mc.contextualizeSlice(sourceField, tag))
			case reflect.Map:
				destField.Set(mc.contextualizeMap(sourceField, tag))
			case reflect.Struct, reflect.Ptr:
				if sourceField.Kind() == reflect.Ptr && sourceField.IsNil() {
					continue
				}
				if tag.Get("type") == "params" {
					destField.Set(reflect.ValueOf(mc.contextualizeStruct(sourceFieldT)))
				} else {
					destField.Set(sourceField)
				}
			default:
				destField.Set(sourceField)
			}
		}
	}

	if !sourceIsPointer {
		return dest.Interface()
	}
	return destPointer.Interface()
}

func (mc modelContextualizer) contextualizeParam(sourceField Param) Param {
	return sourceField.(*param).wrappedParam(mc.graph)
}

func (mc modelContextualizer) contextualizeParamSlice(sourceField []Param) []Param {
	result := make([]Param, len(sourceField))
	for i := 0; i < len(sourceField); i++ {
		result[i] = mc.contextualizeParam(sourceField[i])
	}
	return result
}

func (mc modelContextualizer) contextualizeSlice(sourceField reflect.Value, tag reflect.StructTag) reflect.Value {
	if tag.Get("type") != "params" {
		return sourceField
	}

	length := sourceField.Len()
	result := reflect.MakeSlice(sourceField.Type(), length, length)

	for i := 0; i < length; i++ {
		sourceItem := sourceField.Index(i)

		switch sourceItem.Kind() {
		case reflect.Struct, reflect.Ptr:
			result.Index(i).Set(reflect.ValueOf(mc.contextualizeStruct(sourceItem.Interface())))
		default:
			panic(`nn: "params"-tagged slice contains items with unexpected type`)
		}
	}

	return result
}

var paramInterfaceName = reflect.TypeOf((*Param)(nil)).Elem().Name()

func (mc modelContextualizer) contextualizeMap(sourceValue reflect.Value, tag reflect.StructTag) reflect.Value {
	sourceType := reflect.TypeOf(sourceValue.Interface())
	mapValueType := sourceType.Elem()

	if mapValueType.Name() != paramInterfaceName && tag.Get("type") != "params" {
		return sourceValue
	}

	result := reflect.MakeMap(sourceType)
	mapValueKind := mapValueType.Kind()
	mapRange := sourceValue.MapRange()

	for mapRange.Next() {
		key := mapRange.Key()
		sourceValue := mapRange.Value()

		var destValue reflect.Value

		if p, isParam := sourceValue.Interface().(*param); isParam {
			destValue = reflect.ValueOf(mc.contextualizeParam(p))
		} else if mapValueKind == reflect.Struct || mapValueKind == reflect.Ptr {
			destValue = reflect.ValueOf(mc.contextualizeStruct(sourceValue.Interface()))
		} else {
			panic(`nn: "params"-tagged map contains values with unexpected type`)
		}

		result.SetMapIndex(key, destValue)
	}

	return result
}
