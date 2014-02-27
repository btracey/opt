package multivariate

import (
	"testing"
)

func TestBfgs(t *testing.T) {
	l := NewBfgs()
	GradBasedTest(t, l)
}
