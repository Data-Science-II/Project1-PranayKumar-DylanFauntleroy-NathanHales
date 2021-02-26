//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Dylan Fauntleroy, Nathan Hales, Pranay Kumar, and some repeated code from John Miller
 *  @version 1.6
 *  @date    2/24/2021
 *  @see     LICENSE (MIT style license file).
 *
 *  @title   Project1 Code
 */

package scalation.analytics

import scala.collection.mutable.Set
import scala.util.control.Breaks.{break, breakable}

import scalation.linalgebra._
import scalation.math.noDouble
import scalation.plot.{Plot, PlotM}
import scalation.random.CDF.studentTCDF
import scalation.stat.Statistic
import scalation.stat.StatVector.corr
import scalation.util.banner
import scalation.util.Unicode.sub

import Fit._
import RegTechnique._

//RegTest27 Tests forward selection with MLR for bikes
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RegressionTest27` object tests the `Regression` class using the Bikes
 *  dataset.  It illustrates using the `Relation` class for reading the data
 *  from a .csv file "SeoulBikes.csv".  Assumes no missing values.
 *  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 bar and R^2 cv vs. the instance index.
 *  > runMain scalation.analytics.RegressionTest27
 */
object RegressionTest27 extends App
{
    //Bikes code for MLR
    import scalation.columnar_db.Relation

    banner ("Bikes relation")
    val auto_tab = Relation (BASE_DIR + "SeoulBikes.csv", "Bikes", null, -1)
    //val auto_tab = Relation.apply(BASE_DIR + "SeoulBikeData.csv", "Bikes", 0,  null,  ",", null)
    //auto_tab.show ()

   //var xy = MatrixD(BASE_DIR + "SeoulBikes.csv")
   //val y = xy.col(0)
   //xy = xy.sliceCol(0,9)

    banner ("Bikes (x, y) dataset")
    val (x, y) = auto_tab.toMatriDD (1 to 9, 0)
    println (s"y = $y")
    println (s"x = $x")

    banner ("Bikes regression")
    val rg = new Regression (x, y)
    println (rg.analyze ().report)
    println (rg.summary)

    val (cols, rSq) = rg.forwardSelAll ()                          // R^2, R^2 bar, R^2 cv
    val k = cols.size
	println("k:")
	println(k)
	
    println (s"k = $k, n = ${x.dim2}")
    val t = VectorD.range (1, k)                                   // instance index
    new PlotM (t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"),
               "R^2 vs n for Regression", lines = true)

    println (s"rSq = $rSq")


} // RegressionTest27 object







//RegTest28 Tests backward elimination with MLR for bikes
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RegressionTest28` object tests the `Regression` class using the Bikes
 *  dataset.  It illustrates using the `Relation` class for reading the data
 *  from a .csv file "SeoulBikes.csv".  Assumes no missing values.
 *  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 bar and R^2 cv vs. the instance index.
 *  > runMain scalation.analytics.RegressionTest28
 */
object RegressionTest28 extends App
{
    //Bikes code for MLR
    import scalation.columnar_db.Relation

    banner ("Bikes relation")
    val auto_tab = Relation (BASE_DIR + "SeoulBikes.csv", "Bikes", null, -1)
    //val auto_tab = Relation.apply(BASE_DIR + "SeoulBikes.csv", "Bikes", 0,  null,  ",", null)
    //auto_tab.show ()

    banner ("Bikes (x, y) dataset")
    val (x, y) = auto_tab.toMatriDD (1 to 9, 0)
    println (s"x = $x")
    println (s"y = $y")

    banner ("Bikes regression")
    val rg = new Regression (x, y)
    println (rg.analyze ().report)
    println (rg.summary)

    val (cols, rSq) = rg.backwardElimAll ()                          // R^2, R^2 bar, R^2 cv
    val k = cols.size
	println("k:")
	println(k)
	
    println (s"k = $k, n = ${x.dim2}")
    val t = VectorD.range (1, k)                                   // instance index
    new PlotM (t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"),
               "R^2 vs n for Regression", lines = true)

    println (s"rSq = $rSq")


} // RegressionTest28 object



//RegTest29 Tests stepwise regression with MLR for bikes
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RegressionTest29` object tests the `Regression` class using the Bikes
 *  dataset.  It illustrates using the `Relation` class for reading the data
 *  from a .csv file "SeoulBikes.csv".  Assumes no missing values.
 *  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 bar and R^2 cv vs. the instance index.
 *  > runMain scalation.analytics.RegressionTest29
 */
object RegressionTest29 extends App
{
    //Bikes code for MLR
    import scalation.columnar_db.Relation

    banner ("Bikes relation")
    val auto_tab = Relation (BASE_DIR + "SeoulBikes.csv", "Bikes", null, -1)
    //val auto_tab = Relation.apply(BASE_DIR + "SeoulBikes.csv", "Bikes", 0,  null,  ",", null)
    //auto_tab.show ()

    banner ("Bikes (x, y) dataset")
    val (x, y) = auto_tab.toMatriDD (1 to 9, 0)
    println (s"x = $x")
    println (s"y = $y")

    banner ("Bikes regression")
    val rg = new Regression (x, y)
    println (rg.analyze ().report)
    println (rg.summary)

    val (cols, rSq) = rg.stepRegressionAll ()                          // R^2, R^2 bar, R^2 cv
    val k = cols.size
	println("k:")
	println(k)
	
    println (s"k = $k, n = ${x.dim2}")
    val t = VectorD.range (1, k)                                   // instance index
    new PlotM (t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"),
               "R^2 vs n for Regression", lines = true)

    println (s"rSq = $rSq")


} // RegressionTest29 object


//LassoTest9 tests Lasso regression on MLR for bikes
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `LassoRegressionTest9` object tests the `LassoRegression` class using the Bikes
 *  dataset.  It illustrates using the `Relation` class for reading the data
 *  from a .csv file "SeoulBikes.csv".  Assumes no missing values.
 *  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 Bar and R^2 cv vs. the instance index.
 *  > runMain scalation.analytics.LassoRegressionTest9
 */
object LassoRegressionTest9 extends App
{
    import scalation.columnar_db.Relation

    banner ("Bikes relation")
val auto_tab = Relation (BASE_DIR + "SeoulBikes.csv", "Bikes", null, -1)
    //auto_tab.show ()

    banner ("Bikes (x, y) dataset")
    val (x, y) = auto_tab.toMatriDD (1 to 9, 0)
    println (s"x = $x")
    println (s"y = $y")

    banner ("Bikes regression")
    val rg = new LassoRegression (x, y)
    println (rg.analyze ().report)
    println (rg.summary)
    val n = x.dim2                                                    // number of parameters/variables

    //banner ("Forward Selection Test")
    //val (cols, rSq) = rg.forwardSelAll ()                          // R^2, R^2 Bar, R^2 cv

    banner ("Forward Selection Test")
    val (cols, rSq) = rg.forwardSelAll ()                          // R^2, R^2 Bar, R^2 cv
    val k = cols.size
    val t = VectorD.range (1, k)                                   // instance index
    new PlotM (t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"),
               "R^2 vs n for LassoRegression", lines = true)
    println (s"rSq = $rSq")

} // LassoRegressionTest9 object


//RidgeReg11 tests Ridge regression on MLR for bikes
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RidgeRegressionTest11` object tests the `RidgeRegression` class using the Bikes
 *  dataset.  It illustrates using the `Relation` class for reading the data
 *  from a .csv file "SeoulBikes.csv".  Assumes no missing values.
 *  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 bar and R^2 cv vs. the instance index.
 *  > runMain scalation.analytics.RidgeRegressionTest11
 */
object RidgeRegressionTest11 extends App
{
    import scalation.columnar_db.Relation

    banner ("Bikes relation")
    val auto_tab = Relation (BASE_DIR + "SeoulBikes.csv", "Bikes", null, -1)
    //auto_tab.show ()

    banner ("Bikes (x, y) dataset")
    val (x, y) = auto_tab.toMatriDD (1 to 9, 0)
    println (s"x = $x")
    println (s"y = $y")

    banner ("Bikes regression")
    val rrg = RidgeRegression (x, y, null, RidgeRegression.hp, Cholesky)
    println (rrg.analyze ().report)
    println (rrg.summary)
    val n = x.dim2                                                     // number of variables

    banner ("Forward Selection Test")
    val (cols, rSq) = rrg.forwardSelAll ()                             // R^2, R^2 bar, R^2 cv
    val k = cols.size
    val t = VectorD.range (1, k)                                       // instance index
    new PlotM (t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"),
               "R^2 vs n for RidgeRegression", lines = true)

    println (s"rSq = $rSq")

} // RidgeRegressionTest11 object


//Quadtest19 tests forward selection on Quadratic regression for bikes
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `QuadRegressionTest19` object tests the `QuadRegression` class using the Bikes
 *  dataset.  It illustrates using the `Relation` class for reading the data
 *  from a .csv file "SeoulBikes.csv".  Assumes no missing values.
 *  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 bar and R^2 cv vs. the instance index.
 *  > runMain scalation.analytics.QuadRegressionTest19
 */
object QuadRegressionTest19 extends App
{
    import scalation.columnar_db.Relation

    banner ("Bikes relation")
    val auto_tab = Relation (BASE_DIR + "SeoulBikes.csv", "Bikes", null, -1)
    //auto_tab.show ()

    banner ("Bikes (x, y) dataset")
    val (x, y) = auto_tab.toMatriDD (1 to 9, 0)
    println (s"x = $x")
    println (s"y = $y")

    banner ("Bikes: QuadRegression")
    val qrg = new QuadRegression (x, y)
    println (qrg.analyze ().report)
    val n  = x.dim2                                                  // number of variables
    val nt = QuadRegression.numTerms (n)                             // number of terms
    println (qrg.summary)
    println (s"n = $n, nt = $nt")
    
    banner ("Forward Selection Test")
    val (cols, rSq) = qrg.forwardSelAll ()  
    val k = cols.size
    val t = VectorD.range (1, k)                                     // instance index
    new PlotM (t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"),
               "R^2 vs n for QuadRegression", lines = true)
    println (s"rSq = $rSq")

} // QuadRegressionTest19 object



//Quadtest20 tests backward elimination on Quadratic regression for bikes
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `QuadRegressionTest20` object tests the `QuadRegression` class using the Bikes
 *  dataset.  It illustrates using the `Relation` class for reading the data
 *  from a .csv file "SeoulBikes.csv".  Assumes no missing values.
 *  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 bar and R^2 cv vs. the instance index.
 *  > runMain scalation.analytics.QuadRegressionTest20
 */
object QuadRegressionTest20 extends App
{
    import scalation.columnar_db.Relation

   

    banner ("Bikes relation")
    val auto_tab = Relation (BASE_DIR + "SeoulBikes.csv", "Bikes", null, -1)
    //auto_tab.show ()

    banner ("Bikes (x, y) dataset")
    val (x, y) = auto_tab.toMatriDD (1 to 9, 0)
    println (s"x = $x")
    println (s"y = $y")

    banner ("Bikes: QuadRegression")
    val qrg = new QuadRegression (x, y)
    println (qrg.analyze ().report)
    val n  = x.dim2                                                  // number of variables
    val nt = QuadRegression.numTerms (n)                             // number of terms
    println (qrg.summary)
    println (s"n = $n, nt = $nt")


    banner ("Backward Elimination Test")
    val (cols, rSq) = qrg.backwardElimAll ()                           // R^2, R^2 bar, R^2 cv

    val k = cols.size
    val t = VectorD.range (1, k)                                     // instance index
    new PlotM (t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"),
               "R^2 vs n for QuadRegression", lines = true)
    println (s"rSq = $rSq")

} // QuadRegressionTest20 object


//Quadtest21 tests stepwise regression on Quadratic regression for bikes
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `QuadRegressionTest21` object tests the `QuadRegression` class using the Bikes
 *  dataset.  It illustrates using the `Relation` class for reading the data
 *  from a .csv file "SeoulBikes.csv".  Assumes no missing values.
 *  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 bar and R^2 cv vs. the instance index.
 *  > runMain scalation.analytics.QuadRegressionTest21
 */
object QuadRegressionTest21 extends App
{
    import scalation.columnar_db.Relation

    banner ("Bikes relation")
    val auto_tab = Relation (BASE_DIR + "SeoulBikes.csv", "Bikes", null, -1)
    //auto_tab.show ()

    banner ("Bikes (x, y) dataset")
    val (x, y) = auto_tab.toMatriDD (1 to 9, 0)
    println (s"x = $x")
    println (s"y = $y")

    banner ("Bikes: QuadRegression")
    val qrg = new QuadRegression (x, y)
    println (qrg.analyze ().report)
    val n  = x.dim2                                                  // number of variables
    val nt = QuadRegression.numTerms (n)                             // number of terms
    println (qrg.summary)
    println (s"n = $n, nt = $nt")

    banner ("Stepwise Regression Test")
    val (cols, rSq) = qrg.stepRegressionAll ()                           // R^2, R^2 bar, R^2 cv
    val k = cols.size
    val t = VectorD.range (1, k)                                     // instance index
    new PlotM (t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"),
               "R^2 vs n for QuadRegression", lines = true)
    println (s"rSq = $rSq")

} // QuadRegressionTest21 object


//QuadXTest17 tests QuadX regression using forward selection on the bikes set
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `QuadXRegressionTest17` object tests the `QuadXRegression` class using the Bikes
 *  dataset.  It illustrates using the `Relation` class for reading the data
 *  from a .csv file "SeoulBikes.csv".  Assumes no missing values.
 *  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 bar and R^2 cv vs. the instance index.
 *  > runMain scalation.analytics.QuadXRegressionTest17
 */
object QuadXRegressionTest17 extends App
{
    import scalation.columnar_db.Relation

    banner ("Bikes relation")
    val auto_tab = Relation (BASE_DIR + "SeoulBikes.csv", "Bikes", null, -1)
    //auto_tab.show ()

    banner ("Bikes (x, y) dataset")
    val (x, y) = auto_tab.toMatriDD (1 to 9, 0)
    println (s"x = $x")
    println (s"y = $y")

    banner ("Bikes regression")
    val qrg = new QuadXRegression (x, y)
    println (qrg.analyze ().report)
    val n  = x.dim2                                                // number of variables
    val nt = QuadXRegression.numTerms (n)                          // number of terms
    println (s"n = $n, nt = $nt")
    println (qrg.summary)

    banner ("Forward Selection Test")
    val (cols, rSq) = qrg.forwardSelAll ()                         // R^2, R^2 bar, R^2 cv
    val k = cols.size
    val t = VectorD.range (1, k)                                   // instance index
    new PlotM (t, rSq, lines = true)
    println (s"rSq = $rSq")

} // QuadXRegressionTest17 object


//QuadXTest18 tests QuadX regression using backward elimination on the bikes set
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `QuadXRegressionTest18` object tests the `QuadXRegression` class using the Bikes
 *  dataset.  It illustrates using the `Relation` class for reading the data
 *  from a .csv file "SeoulBikes.csv".  Assumes no missing values.
 *  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 bar and R^2 cv vs. the instance index.
 *  > runMain scalation.analytics.QuadXRegressionTest18
 */
object QuadXRegressionTest18 extends App
{
    import scalation.columnar_db.Relation

    banner ("Bikes relation")
    val auto_tab = Relation (BASE_DIR + "SeoulBikes.csv", "Bikes", null, -1)
    //auto_tab.show ()

    banner ("Bikes (x, y) dataset")
    val (x, y) = auto_tab.toMatriDD (1 to 9, 0)
    println (s"x = $x")
    println (s"y = $y")

    banner ("Bikes regression")
    val qrg = new QuadXRegression (x, y)
    println (qrg.analyze ().report)
    val n  = x.dim2                                                // number of variables
    val nt = QuadXRegression.numTerms (n)                          // number of terms
    println (s"n = $n, nt = $nt")
    println (qrg.summary)

    banner ("Backward Elimination Test")
    val (cols, rSq) = qrg.backwardElimAll()    
    val k = cols.size
    val t = VectorD.range (1, k)                                   // instance index
    new PlotM (t, rSq, lines = true)
    println (s"rSq = $rSq")

} // QuadXRegressionTest18 object


//QuadXTest19 tests QuadX regression using stepwise regression on the bikes set
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `QuadXRegressionTest19` object tests the `QuadXRegression` class using the Bikes
 *  dataset.  It illustrates using the `Relation` class for reading the data
 *  from a .csv file "SeoulBikes.csv".  Assumes no missing values.
 *  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 bar and R^2 cv vs. the instance index.
 *  > runMain scalation.analytics.QuadXRegressionTest19
 */
object QuadXRegressionTest19 extends App
{
    import scalation.columnar_db.Relation

     banner ("Bikes relation")
    val auto_tab = Relation (BASE_DIR + "SeoulBikes.csv", "Bikes", null, -1)
    //auto_tab.show ()

    banner ("Bikes (x, y) dataset")
    val (x, y) = auto_tab.toMatriDD (1 to 9, 0)
    println (s"x = $x")
    println (s"y = $y")

    banner ("Bikes regression")
    val qrg = new QuadXRegression (x, y)
    println (qrg.analyze ().report)
    val n  = x.dim2                                                // number of variables
    val nt = QuadXRegression.numTerms (n)                          // number of terms
    println (s"n = $n, nt = $nt")
    println (qrg.summary)

    banner ("Stepwise Regression Test")
    val (cols, rSq) = qrg.stepRegressionAll()    
    val k = cols.size
    val t = VectorD.range (1, k)                                   // instance index
    new PlotM (t, rSq, lines = true)
    println (s"rSq = $rSq")

} // QuadXRegressionTest19 object




//CubicTest17 tests cubic regression using forward selection on bikes
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `CubicRegressionTest17` object tests the `CubicRegression` class using the Bikes
 *  dataset.  It illustrates using the `Relation` class for reading the data
 *  from a .csv file "SeoulBikes.csv".  Assumes no missing values.
 *  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 bar and R^2 cv vs. the instance index.
 *  > runMain scalation.analytics.CubicRegressionTest17
 */
object CubicRegressionTest17 extends App
{
    import scalation.columnar_db.Relation

    banner ("Bikes relation")
    val auto_tab = Relation (BASE_DIR + "SeoulBikes.csv", "Bikes", null, -1)
    //auto_tab.show ()

    banner ("Bikes (x, y) dataset")
    val (x, y) = auto_tab.toMatriDD (1 to 9, 0)

//  import ExampleBikes.{x, y}

    println (s"x = $x")
    println (s"y = $y")

    banner ("Bikes regression")
    val crg = new CubicRegression (x, y)
    println (crg.analyze ().report)

    val n  = x.dim2                                                // number of variables
    val nt = CubicRegression.numTerms (n)                          // number of terms
    println (s"n = $n, nt = $nt")
    println (crg.summary)

    banner ("Forward Selection Test")
    val (cols, rSq) = crg.forwardSelAll ()                         // R^2, R^2 bar, R^2 cv

    val k = cols.size - 1
    val t = VectorD.range (1, k)                                   // instance index
    new PlotM (t, rSq.t, lines = true)
    println (s"k = $k, nt = $nt")
    println (s"rSq = $rSq")

} // CubicRegressionTest17 object




//CubicTest18 tests cubic regression using backward elimination on bikes
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `CubicRegressionTest18` object tests the `CubicRegression` class using the Bikes
 *  dataset.  It illustrates using the `Relation` class for reading the data
 *  from a .csv file "SeoulBikes.csv".  Assumes no missing values.
 *  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 bar and R^2 cv vs. the instance index.
 *  > runMain scalation.analytics.CubicRegressionTest18
 */
object CubicRegressionTest18 extends App
{
    import scalation.columnar_db.Relation

    banner ("Bikes relation")
    val auto_tab = Relation (BASE_DIR + "SeoulBikes.csv", "Bikes", null, -1)
    //auto_tab.show ()

    banner ("Bikes (x, y) dataset")
    val (x, y) = auto_tab.toMatriDD (1 to 9, 0)

//  import ExampleBikes.{x, y}

    println (s"x = $x")
    println (s"y = $y")

    banner ("Bikes regression")
    val crg = new CubicRegression (x, y)
    println (crg.analyze ().report)

    val n  = x.dim2                                                // number of variables
    val nt = CubicRegression.numTerms (n)                          // number of terms
    println (s"n = $n, nt = $nt")
    println (crg.summary)

    banner ("Backward Elimination Test")
    val (cols, rSq) = crg.backwardElimAll ()                         // R^2, R^2 bar, R^2 cv

    val k = cols.size - 1
    val t = VectorD.range (1, k)                                   // instance index
    new PlotM (t, rSq.t, lines = true)
    println (s"k = $k, nt = $nt")
    println (s"rSq = $rSq")

} // CubicRegressionTest18 object

//CubicTest19 tests cubic regression using stepwise regression on bikes
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `CubicRegressionTest19` object tests the `CubicRegression` class using the Bikes
 *  dataset.  It illustrates using the `Relation` class for reading the data
 *  from a .csv file "SeoulBikes.csv".  Assumes no missing values.
 *  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 bar and R^2 cv vs. the instance index.
 *  > runMain scalation.analytics.CubicRegressionTest19
 */
object CubicRegressionTest19 extends App
{
    import scalation.columnar_db.Relation

    banner ("Bikes relation")
    val auto_tab = Relation (BASE_DIR + "SeoulBikes.csv", "Bikes", null, -1)
    //auto_tab.show ()

    banner ("Bikes (x, y) dataset")
    val (x, y) = auto_tab.toMatriDD (1 to 9, 0)

//  import ExampleBikes.{x, y}

    println (s"x = $x")
    println (s"y = $y")

    banner ("Bikes regression")
    val crg = new CubicRegression (x, y)
    println (crg.analyze ().report)

    val n  = x.dim2                                                // number of variables
    val nt = CubicRegression.numTerms (n)                          // number of terms
    println (s"n = $n, nt = $nt")
    println (crg.summary)


    banner ("Stepwise Regression Test")
    val (cols, rSq) = crg.stepRegressionAll () 
    val k = cols.size - 1
    val t = VectorD.range (1, k)                                   // instance index
    new PlotM (t, rSq.t, lines = true)
    println (s"k = $k, nt = $nt")
    println (s"rSq = $rSq")

} // CubicRegressionTest19 object



//cubicXtest17 tests forward selection on cubicX regression for bikes
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `CubicXRegressionTest17` object tests the `CubicXRegression` class using the Bikes
 *  dataset.  It illustrates using the `Relation` class for reading the data
 *  from a .csv file "SeoulBikes.csv".  Assumes no missing values.
 *  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 bar and R^2 cv vs. the instance index.
 *  > runMain scalation.analytics.CubicXRegressionTest17
 */
object CubicXRegressionTest17 extends App
{
    import scalation.columnar_db.Relation

    banner ("Bikes relation")
    val auto_tab = Relation (BASE_DIR + "SeoulBikes.csv", "Bikes", null, -1)
    //auto_tab.show ()

    banner ("Bikes (x, y) dataset")
    val (x, y) = auto_tab.toMatriDD (1 to 9, 0)
    println (s"x = $x")
    println (s"y = $y")

    banner ("Bikes regression")
    val crg = new CubicXRegression (x, y)
    println (crg.analyze ().report)

    val n  = x.dim2                                                // number of variables
    val nt = CubicXRegression.numTerms (n)                          // number of terms
    println (s"n = $n, nt = $nt")
    println (crg.summary)

    banner ("Forward Selection Test")
    
    val (cols, rSq) = crg.forwardSelAll ()                          // R^2, R^2 bar, R^2 cv
   

    val k = cols.size
    val t = VectorD.range (1, k)                                   // instance index
    new PlotM (t, rSq, lines = true)
    println (s"rSq = $rSq")

} // CubicXRegressionTest17 object



//cubicXtest18 tests backward elimination on cubicX regression for bikes
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `CubicXRegressionTest18` object tests the `CubicXRegression` class using the Bikes
 *  dataset.  It illustrates using the `Relation` class for reading the data
 *  from a .csv file "SeoulBikes.csv".  Assumes no missing values.
 *  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 bar and R^2 cv vs. the instance index.
 *  > runMain scalation.analytics.CubicXRegressionTest18
 */
object CubicXRegressionTest18 extends App
{
    import scalation.columnar_db.Relation

    banner ("Bikes relation")
    val auto_tab = Relation (BASE_DIR + "SeoulBikes.csv", "Bikes", null, -1)
    //auto_tab.show ()

    banner ("Bikes (x, y) dataset")
    val (x, y) = auto_tab.toMatriDD (1 to 9, 0)
    println (s"x = $x")
    println (s"y = $y")

    banner ("Bikes regression")
    val crg = new CubicXRegression (x, y)
    println (crg.analyze ().report)

    val n  = x.dim2                                                // number of variables
    val nt = CubicXRegression.numTerms (n)                          // number of terms
    println (s"n = $n, nt = $nt")
    println (crg.summary)



    banner ("Backward Elimination Test")
    val (cols, rSq) = crg.backwardElimAll ()      

    val k = cols.size
    val t = VectorD.range (1, k)                                   // instance index
    new PlotM (t, rSq, lines = true)
    println (s"rSq = $rSq")

} // CubicXRegressionTest18 object



//cubicXtest19 tests stepwise regression on cubicX regression for bikes
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `CubicXRegressionTest19` object tests the `CubicXRegression` class using the Bikes
 *  dataset.  It illustrates using the `Relation` class for reading the data
 *  from a .csv file "SeoulBikes.csv".  Assumes no missing values.
 *  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 bar and R^2 cv vs. the instance index.
 *  > runMain scalation.analytics.CubicXRegressionTest19
 */
object CubicXRegressionTest19 extends App
{
    import scalation.columnar_db.Relation

    banner ("Bikes relation")
    val auto_tab = Relation (BASE_DIR + "SeoulBikes.csv", "Bikes", null, -1)
    //auto_tab.show ()

    banner ("Bikes (x, y) dataset")
    val (x, y) = auto_tab.toMatriDD (1 to 9, 0)
    println (s"x = $x")
    println (s"y = $y")

    banner ("Bikes regression")
    val crg = new CubicXRegression (x, y)
    println (crg.analyze ().report)

    val n  = x.dim2                                                // number of variables
    val nt = CubicXRegression.numTerms (n)                          // number of terms
    println (s"n = $n, nt = $nt")
    println (crg.summary)


    banner ("Stepwise Regression Test")
    val (cols, rSq) = crg.stepRegressionAll ()      

    val k = cols.size
    val t = VectorD.range (1, k)                                   // instance index
    new PlotM (t, rSq, lines = true)
    println (s"rSq = $rSq")

} // CubicXRegressionTest19 object

