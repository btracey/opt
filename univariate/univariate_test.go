package univariate

type GradFreeTestFunction struct {
	f        func(float64) float64
	initLocs []float64
	optLocs  []float64
	optVals  []float64
	Name     string
}

type quadratic struct {
	b float64
	c float64
}

func (q quadratic) Obj(x float64) float64 {
	return (x-q.b)*(x-q.b) + q.c
}

func (q quadratic) OptVal() float64 {
	return q.c
}

func (q quadratic) OptLoc() float64 {
	return q.b
}

func (q quadratic) Convex() {}
