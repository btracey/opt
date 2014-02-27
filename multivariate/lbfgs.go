package multivariate

import (
	"errors"
	"github.com/btracey/opt/common"
	"github.com/btracey/opt/multivariate/linesearch"

	"github.com/gonum/blas/dbw"
	"github.com/gonum/floats"
)

type Lbfgs struct {
	LinesearchSettings *linesearch.Settings
	Memory             int // How many past iterations

	fun  ObjGrader
	nDim int

	counter int // Counter is where the new values will be stored
	looped  bool

	// Other needed variables
	step    float64
	gamma_k float64

	q     dbw.Vector // current direction
	alpha []float64
	//rhoHist    []float64
	invRhoHist []float64
	sHist      [][]float64
	yHist      [][]float64

	currGrad []float64
	currLoc  []float64
	currObj  float64
	prevObj  float64
}

func NewLbfgs() *Lbfgs {
	return &Lbfgs{
		LinesearchSettings: linesearch.DefaultSettings(),
		Memory:             30,
	}
}

func (lbfgs *Lbfgs) Init(f ObjGrader, initLoc []float64, initObj float64, initGrad []float64) error {
	if initLoc == nil {
		return errors.New("lbfgs: initLoc is nil")
	}
	if initGrad == nil {
		return errors.New("lbfgs: initGrad is nil")
	}

	lbfgs.fun = f
	lbfgs.nDim = len(initLoc)

	lbfgs.counter = 0
	lbfgs.looped = false

	lbfgs.q = dbw.NewVector(make([]float64, lbfgs.nDim))
	lbfgs.alpha = make([]float64, lbfgs.Memory)
	//lbfgs.rhoHist = make([]float64, lbfgs.Memory)
	lbfgs.invRhoHist = make([]float64, lbfgs.Memory)
	lbfgs.sHist = make([][]float64, lbfgs.Memory)
	lbfgs.yHist = make([][]float64, lbfgs.Memory)
	for i := range lbfgs.sHist {
		lbfgs.sHist[i] = make([]float64, lbfgs.nDim)
		lbfgs.yHist[i] = make([]float64, lbfgs.nDim)
	}

	lbfgs.currGrad = make([]float64, lbfgs.nDim)
	lbfgs.currLoc = make([]float64, lbfgs.nDim)
	copy(lbfgs.currGrad, initGrad)
	copy(lbfgs.currLoc, initLoc)
	lbfgs.currObj = initObj
	lbfgs.prevObj = initObj + 5000 // trick from scipy

	copy(lbfgs.q.Data, initGrad)
	dbw.Scal(-1, lbfgs.q)
	return nil

}

func (lbfgs *Lbfgs) Status() common.Status {
	return common.Continue
}

func (lbfgs *Lbfgs) Next(loc []float64) {
	copy(loc, lbfgs.q.Data)
	floats.Add(loc, lbfgs.currLoc)
}

// TODO: Clean up vector and make all use blas

// New loc and grad are stored in place
func (lbfgs *Lbfgs) Iterate(loc, grad []float64) (obj float64, nFunEvals int, err error) {
	counter := lbfgs.counter
	m := lbfgs.Memory

	if len(loc) != lbfgs.nDim {
		panic("dimension mismatch")
	}
	if len(grad) != lbfgs.nDim {
		panic("dimension mismatch")
	}

	//fmt.Println("in lbfgs grad norm is ", floats.Norm(lbfgs.currGrad, 2))
	/*
		fmt.Println("lbfgs search vector ", lbfgs.q.Data)
		fmt.Println("lbfgs currGrad ", lbfgs.currGrad)
		fmt.Println("lbfgs currLoc", lbfgs.currLoc)
	*/
	result, err := linesearch.GradLinesearch(lbfgs.LinesearchSettings, lbfgs.fun,
		lbfgs.q.Data, lbfgs.currLoc, lbfgs.currObj, lbfgs.currGrad, lbfgs.prevObj)

	if err != nil {
		return 0, 0, err
	}
	/*
		fmt.Println("lbfgs search vector ", lbfgs.q.Data)
		fmt.Println("lbfgs new grad", result.Grad)
		fmt.Println("lbfgs newLoc", result.Loc)
	*/
	copy(loc, result.Loc)
	copy(grad, result.Grad)

	newLoc := result.Loc
	newGrad := result.Grad

	// y_k = g_{k+1} - g_k
	copy(lbfgs.yHist[counter], newGrad)
	floats.Sub(lbfgs.yHist[counter], lbfgs.currGrad)

	// s_k = x_{k+1} - x_k
	copy(lbfgs.sHist[counter], newLoc)
	floats.Sub(lbfgs.sHist[counter], lbfgs.currLoc)

	invRho := floats.Dot(lbfgs.yHist[counter], lbfgs.sHist[counter])
	lbfgs.invRhoHist[counter] = invRho

	// Calculate the search direction for the new iteration
	// q = gk
	copy(lbfgs.q.Data, newGrad)
	floats.Scale(-1, lbfgs.q.Data)
	// Update based on the previous gradient values (most recent first)

	max := m
	if !lbfgs.looped {
		max = counter + 1 // Can't go m iterations ago
	}

	// i is how many iterations ago (zero indexed because the linesearch found k+1)
	for i := 0; i < max; i++ {
		//for i := m - 1; i >= 0; i-- {
		ind := counter - i // go back i iterations
		if ind < 0 {       // storage is wrapped
			ind += m
		}
		//fmt.Println("ind", ind, "rho ind", lbfgs.rhoHist[ind], "s dot q", floats.Dot(lbfgs.sHist[ind], lbfgs.q.Data))

		// TODO: Check here to see if we can improve the numerical scaling

		//fmt.Println("shist", lbfgs.sHist[ind], "invrho", lbfgs.invRhoHist[ind])
		lbfgs.alpha[ind] = floats.Dot(lbfgs.sHist[ind], lbfgs.q.Data) / lbfgs.invRhoHist[ind]
		yVec := dbw.NewVector(lbfgs.yHist[ind])
		dbw.Axpy(-1*lbfgs.alpha[ind], yVec, lbfgs.q)
	}

	gamma_k := floats.Dot(lbfgs.yHist[counter], lbfgs.sHist[counter])
	gamma_k /= floats.Dot(lbfgs.yHist[counter], lbfgs.yHist[counter])

	//fmt.Println("gamma k", gamma_k)

	//	fmt.Println("gamma k = ", gamma_k)

	//dbw.Scal(-gamma_k, lbfgs.q)

	//for i := 0; i < m; i++ {
	for i := max - 1; i >= 0; i-- {
		ind := counter - i
		if ind < 0 {
			ind += m
		}
		//fmt.Println("ind2", ind)
		beta := floats.Dot(lbfgs.yHist[ind], lbfgs.q.Data) / lbfgs.invRhoHist[ind]
		sVec := dbw.NewVector(lbfgs.sHist[ind])
		dbw.Axpy(lbfgs.alpha[ind]-beta, sVec, lbfgs.q)
	}
	//dbw.Scal(-1, lbfgs.q)

	/*
		fmt.Println()
		fmt.Println("lbfgs q ", lbfgs.q)
		fmt.Println("lbfgs grad ", newGrad)
		fmt.Println("lbfgs grad norm", floats.Norm(newGrad, 2))
		fmt.Println("rho hist", lbfgs.rhoHist)
		fmt.Println("alhpha hist", lbfgs.alpha)
		fmt.Println("y hist", lbfgs.yHist)
	*/
	lbfgs.counter++
	if lbfgs.counter == lbfgs.Memory {
		lbfgs.counter = 0
		lbfgs.looped = true
	}

	lbfgs.prevObj = lbfgs.currObj
	lbfgs.currObj = result.Obj
	copy(lbfgs.currLoc, result.Loc)
	copy(lbfgs.currGrad, result.Grad)
	copy(loc, result.Loc)
	copy(grad, result.Grad)

	//grdNrm := floats.Norm(grad, 2)

	return result.Obj, result.NFunEvals, nil
}

func (lbfgs *Lbfgs) Result() {}
