package multivariate

import (
	"github.com/btracey/opt/common"
	"github.com/btracey/opt/multivariate/linesearch"

	"github.com/gonum/floats"
	"github.com/gonum/matrix/mat64"
)

type Bfgs struct {
	LinesearchSettings *linesearch.Settings

	InitialInverseHessian *mat64.Dense
	tmpMat                *mat64.Dense

	fun     ObjGrader
	nDim    int
	invHess *mat64.Dense

	currLoc  []float64
	currObj  float64
	prevObj  float64
	currGrad []float64
	p        []float64 // Step direction
	s        []float64
	y        []float64
}

func NewBfgs() *Bfgs {
	return &Bfgs{
		LinesearchSettings: linesearch.DefaultSettings(),
	}
}

func (bfgs *Bfgs) Init(f ObjGrader, initLoc []float64, initObj float64, initGrad []float64) error {
	// TODO: Add error checking
	bfgs.fun = f
	bfgs.nDim = len(initLoc)

	bfgs.currLoc = make([]float64, bfgs.nDim)
	copy(bfgs.currLoc, initLoc)
	bfgs.currGrad = make([]float64, bfgs.nDim)
	copy(bfgs.currGrad, initGrad)
	bfgs.currObj = initObj
	bfgs.prevObj = initObj + 5000 // trick taken from scipy

	bfgs.y = make([]float64, bfgs.nDim)
	bfgs.s = make([]float64, bfgs.nDim)

	// initialize inv hessian to the identity matrix
	// TODO: Add initial hessian jazz
	bfgs.invHess = mat64.NewDense(bfgs.nDim, bfgs.nDim, nil)
	for i := 0; i < bfgs.nDim; i++ {
		bfgs.invHess.Set(i, i, 1)
	}
	bfgs.tmpMat = mat64.NewDense(bfgs.nDim, bfgs.nDim, nil)

	bfgs.p = make([]float64, bfgs.nDim)
	pmat := mat64.NewDense(bfgs.nDim, 1, bfgs.p)
	gradMat := mat64.NewDense(bfgs.nDim, 1, initGrad)

	pmat.Mul(bfgs.invHess, gradMat)
	floats.Scale(-1, bfgs.p)

	return nil
}

func (bfgs *Bfgs) Status() common.Status {
	return common.Continue
}

func (bfgs *Bfgs) Next(loc []float64) {
	copy(loc, bfgs.currLoc)
	floats.Add(loc, bfgs.p)
}

func (bfgs *Bfgs) Iterate(loc, grad []float64) (obj float64, nFunEvals int, err error) {
	if len(loc) != bfgs.nDim {
		panic("dimension mismatch")
	}
	if len(grad) != bfgs.nDim {
		panic("dimension mismatch")
	}
	result, err := linesearch.GradLinesearch(bfgs.LinesearchSettings, bfgs.fun,
		bfgs.p, bfgs.currLoc, bfgs.currObj, bfgs.currGrad, bfgs.prevObj)

	// TODO: Improve this error checking
	if err != nil {
		return 0, 0, err
	}

	newLoc := result.Loc
	newGrad := result.Grad

	// y_k = g_{k+1} - g_k
	copy(bfgs.y, newGrad)
	floats.Sub(bfgs.y, bfgs.currGrad)

	// Compute the recent step
	// s_k = x_{k+1} - x_k
	copy(bfgs.s, newLoc)
	floats.Sub(bfgs.s, bfgs.currLoc)

	// Compute sk^T yk
	sk_yk := floats.Dot(bfgs.y, bfgs.s)
	sk_yk_Sq := sk_yk * sk_yk

	ykMat := mat64.NewDense(bfgs.nDim, 1, bfgs.y)
	ykTmat := mat64.NewDense(1, bfgs.nDim, bfgs.y)

	skMat := mat64.NewDense(bfgs.nDim, 1, bfgs.s)
	skTmat := mat64.NewDense(1, bfgs.nDim, bfgs.s)

	tmp1 := mat64.NewDense(0, 0, nil)
	tmp1.Mul(ykTmat, bfgs.invHess)
	tmp1.Mul(tmp1, ykMat)

	r, c := tmp1.Dims()
	if r != 1 || c != 1 {
		panic("wrong dims")
	}
	val := (sk_yk + tmp1.At(0, 0)) / (sk_yk_Sq)

	tmp1.Reset() // to reuse for multiplication
	tmp1.Mul(skMat, skTmat)
	tmp1.Scale(val, tmp1)

	tmp2 := mat64.NewDense(0, 0, nil)
	tmp2.Mul(bfgs.invHess, ykMat)
	tmp2.Mul(tmp2, skTmat)
	tmp2.Scale(-1/sk_yk, tmp2)

	tmp3 := mat64.NewDense(0, 0, nil)
	tmp3.Mul(ykTmat, bfgs.invHess)
	tmp3.Mul(skMat, tmp3)
	tmp3.Scale(-1/sk_yk, tmp3)

	// Update inverse hessian
	bfgs.invHess.Add(bfgs.invHess, tmp1)
	bfgs.invHess.Add(bfgs.invHess, tmp2)
	bfgs.invHess.Add(bfgs.invHess, tmp3)

	// Update the current location and gradient
	copy(bfgs.currGrad, newGrad)
	copy(bfgs.currLoc, newLoc)
	bfgs.prevObj = bfgs.currObj
	bfgs.currObj = result.Obj

	// Find a new search direction
	dirmat := mat64.NewDense(bfgs.nDim, 1, bfgs.p)
	gradmat := mat64.NewDense(bfgs.nDim, 1, bfgs.currGrad)
	dirmat.Mul(bfgs.invHess, gradmat) // no copy needed because underlying matrix is modified
	floats.Scale(-1, bfgs.p)

	//	fmt.Println("invHess", bfgs.invHess)

	// Copy information to output
	copy(loc, newLoc)
	copy(grad, newGrad)
	return result.Obj, result.NFunEvals, nil
}

func (bfgs *Bfgs) Result() {
	// TODO: Expose final hessian estimate
	return
}
