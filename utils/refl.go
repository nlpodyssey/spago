// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package utils

import "reflect"

// ForEachField calls the callback for each field of the struct i.
func ForEachField(i any, callback func(field any, name string, tag reflect.StructTag)) {
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
			callback(vField.Interface(), tField.Name, tField.Tag)
		}
	}
}
