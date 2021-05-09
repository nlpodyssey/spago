// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	"fmt"
	"github.com/awalterschulze/gographviz"
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"reflect"
	"strconv"
	"strings"
)

// GraphvizGraph creates a gographviz graph representation of the Graph.
func (g *Graph) GraphvizGraph() (gographviz.Interface, error) {
	gg := gographviz.NewEscape()

	if err := gg.SetDir(true); err != nil {
		return nil, err
	}

	for _, node := range g.nodes {
		switch nt := node.(type) {
		case *variable:
			if err := g.addGVVariable(gg, nt); err != nil {
				return nil, err
			}
		case *operator:
			if err := g.addGVOperator(gg, nt); err != nil {
				return nil, err
			}
		// TODO: case *param
		// TODO: case *wrappedParam
		// TODO: case *wrapper
		default:
			return nil, fmt.Errorf("unexpected node type %T", node)
		}
	}
	return gg, nil
}

func (g *Graph) addGVVariable(gg gographviz.Interface, v *variable) error {
	id := fmt.Sprintf("%d", v.ID())
	label := fmt.Sprintf(
		`<
			<TABLE BORDER="0">
				<TR><TD><FONT COLOR="#707070" POINT-SIZE="11">%d</FONT></TD></TR>
				<TR><TD>variable</TD></TR>
				<TR><TD><FONT FACE="monospace" POINT-SIZE="9">%s</FONT></TD></TR>
			</TABLE>
		>`,
		v.ID(),
		gvMatrixTable(v.value),
	)
	attrs := map[string]string{
		"label": label,
		"shape": "box",
	}
	return gg.AddNode("", id, attrs)
}

func (g *Graph) addGVOperator(gg gographviz.Interface, op *operator) error {
	operatorID := fmt.Sprintf("%d", op.ID())
	funcName := reflect.ValueOf(op.function).Elem().Type().Name()
	label := fmt.Sprintf(
		`<
			<TABLE BORDER="0">
				<TR><TD><FONT COLOR="#707070" POINT-SIZE="11">%d</FONT></TD></TR>
				<TR><TD>operator</TD></TR>
				<TR><TD><B>%s</B></TD></TR>
			</TABLE>
		>`,
		op.ID(),
		funcName,
	)
	attrs := map[string]string{
		"label": label,
	}
	if err := gg.AddNode("", operatorID, attrs); err != nil {
		return err
	}

	for _, operand := range op.operands {
		operandID := fmt.Sprintf("%d", operand.ID())
		if err := gg.AddEdge(operandID, operatorID, true, nil); err != nil {
			return err
		}
	}
	return nil
}

func gvMatrixTable(m mat.Matrix) string {
	var b strings.Builder
	b.WriteString(`<TABLE BORDER="0" CELLSPACING="0">`)

	nRows := m.Rows()
	nCols := m.Columns()

	collapseThreshold := 7
	collapsedRows := nRows > collapseThreshold
	collapsedCols := nCols > collapseThreshold
	nonCollapsed := 3

	b.WriteString(`<TR><TD BORDER="0"></TD>`)
	for c := 0; c < nCols; c++ {
		if collapsedCols && c == nonCollapsed {
			b.WriteString(`<TD BORDER="1"><B>…</B></TD>`)
			c = nCols - nonCollapsed - 1
			continue
		}
		b.WriteString(fmt.Sprintf(`<TD BORDER="1"><B>%d</B></TD>`, c))
	}
	b.WriteString("</TR>")

	for r := 0; r < nRows; r++ {
		if collapsedRows && r == nonCollapsed {
			b.WriteString(`<TR><TD BORDER="1"><B>…</B></TD>`)
			for c := 0; c < nCols; c++ {
				b.WriteString(`<TD BORDER="1"><B>…</B></TD>`)
				if collapsedCols && c == nonCollapsed {
					c = nCols - nonCollapsed - 1
				}
			}
			b.WriteString("</TR>")
			r = nRows - nonCollapsed - 1
			continue
		}

		b.WriteString(fmt.Sprintf(`<TR><TD BORDER="1"><B>%d</B></TD>`, r))
		for c := 0; c < nCols; c++ {
			if collapsedCols && c == nonCollapsed {
				b.WriteString(`<TD BORDER="1"><B>…</B></TD>`)
				c = nCols - nonCollapsed - 1
				continue
			}
			f := strconv.FormatFloat(float64(m.At(r, c)), 'g', -1, 64)
			b.WriteString(fmt.Sprintf(`<TD BORDER="1">%s</TD>`, f))
		}
		b.WriteString("</TR>")
	}

	b.WriteString("</TABLE>")
	return b.String()
}
