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

//RegTest22 Tests forward selection with MLR for stability
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RegressionTest22` object tests the `Regression` class using the Stability
 *  dataset.  It illustrates using the `Relation` class for reading the data
 *  from a .csv file "Stability_Data.csv".  Assumes no missing values.
 *  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 bar and R^2 cv vs. the instance index.
 *  > runMain scalation.analytics.RegressionTest22
 */
object RegressionTest22 extends App
{
    //Stability code for MLR
    import scalation.columnar_db.Relation

    banner ("Stability relation")
    val auto_tab = Relation (BASE_DIR + "Stability_Data.csv", "Stability", null, -1)
    //val auto_tab = Relation.apply(BASE_DIR + "Stability_Data.csv", "Stability", 0,  null,  ",", null)
    //auto_tab.show ()

    banner ("Stability (x, y) dataset")
    val (x, y) = auto_tab.toMatriDD (0 to 11, 12)
    println (s"x = $x")
    println (s"y = $y")

    banner ("Stability regression")
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


} // RegressionTest22 object









//RegTest23 Tests backward elimination with MLR for stability
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RegressionTest23` object tests the `Regression` class using the Stability
 *  dataset.  It illustrates using the `Relation` class for reading the data
 *  from a .csv file "Stability_Data.csv".  Assumes no missing values.
 *  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 bar and R^2 cv vs. the instance index.
 *  > runMain scalation.analytics.RegressionTest23
 */
object RegressionTest23 extends App
{
    //Stability code for MLR
    import scalation.columnar_db.Relation

    banner ("Stability relation")
    val auto_tab = Relation (BASE_DIR + "Stability_Data.csv", "Stability", null, -1)
    //val auto_tab = Relation.apply(BASE_DIR + "Stability_Data.csv", "Stability", 0,  null,  ",", null)
    //auto_tab.show ()

    banner ("Stability (x, y) dataset")
    val (x, y) = auto_tab.toMatriDD (0 to 11, 12)
    println (s"x = $x")
    println (s"y = $y")

    banner ("Stability regression")
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


} // RegressionTest23 object



//RegTest24 Tests stepwise regression with MLR for stability
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RegressionTest24` object tests the `Regression` class using the Stability
 *  dataset.  It illustrates using the `Relation` class for reading the data
 *  from a .csv file "Stability_Data.csv".  Assumes no missing values.
 *  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 bar and R^2 cv vs. the instance index.
 *  > runMain scalation.analytics.RegressionTest24
 */
object RegressionTest24 extends App
{
    //Stability code for MLR
    import scalation.columnar_db.Relation

    banner ("Stability relation")
    val auto_tab = Relation (BASE_DIR + "Stability_Data.csv", "Stability", null, -1)
    //val auto_tab = Relation.apply(BASE_DIR + "Stability_Data.csv", "Stability", 0,  null,  ",", null)
    //auto_tab.show ()

    banner ("Stability (x, y) dataset")
    val (x, y) = auto_tab.toMatriDD (0 to 11, 12)
    println (s"x = $x")
    println (s"y = $y")

    banner ("Stability regression")
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


} // RegressionTest24 object


//LassoTest8 tests Lasso regression on MLR for stability
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `LassoRegressionTest8` object tests the `LassoRegression` class using the Stability
 *  dataset.  It illustrates using the `Relation` class for reading the data
 *  from a .csv file "Stability_Data.csv".  Assumes no missing values.
 *  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 Bar and R^2 cv vs. the instance index.
 *  > runMain scalation.analytics.LassoRegressionTest8
 */
object LassoRegressionTest8 extends App
{
    import scalation.columnar_db.Relation

    banner ("Stability relation")
val auto_tab = Relation (BASE_DIR + "Stability_Data.csv", "Stability", null, -1)
    //auto_tab.show ()

    banner ("Stability (x, y) dataset")
    val (x, y) = auto_tab.toMatriDD (0 to 11, 12)
    println (s"x = $x")
    println (s"y = $y")

    banner ("Stability regression")
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

} // LassoRegressionTest8 object


//RidgeReg10 tests Ridge regression on MLR for stability
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RidgeRegressionTest10` object tests the `RidgeRegression` class using the Stability
 *  dataset.  It illustrates using the `Relation` class for reading the data
 *  from a .csv file "Stability_Data.csv".  Assumes no missing values.
 *  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 bar and R^2 cv vs. the instance index.
 *  > runMain scalation.analytics.RidgeRegressionTest10
 */
object RidgeRegressionTest10 extends App
{
    import scalation.columnar_db.Relation

    banner ("Stability relation")
    val auto_tab = Relation (BASE_DIR + "Stability_Data.csv", "Stability", null, -1)
    //auto_tab.show ()

    banner ("Stability (x, y) dataset")
    val (x, y) = auto_tab.toMatriDD (0 to 11, 12)
    println (s"x = $x")
    println (s"y = $y")

    banner ("Stability regression")
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

} // RidgeRegressionTest10 object


//Quadtest16 tests forward selection on Quadratic regression for stability
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `QuadRegressionTest16` object tests the `QuadRegression` class using the Stability
 *  dataset.  It illustrates using the `Relation` class for reading the data
 *  from a .csv file "Stability_Data.csv".  Assumes no missing values.
 *  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 bar and R^2 cv vs. the instance index.
 *  > runMain scalation.analytics.QuadRegressionTest16
 */
object QuadRegressionTest16 extends App
{
    import scalation.columnar_db.Relation

    banner ("Stability relation")
    val auto_tab = Relation (BASE_DIR + "Stability_Data.csv", "Stability", null, -1)
    //auto_tab.show ()

    banner ("Stability (x, y) dataset")
    val (x, y) = auto_tab.toMatriDD (0 to 11, 12)
    println (s"x = $x")
    println (s"y = $y")

    banner ("Stability: QuadRegression")
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

} // QuadRegressionTest16 object



//Quadtest17 tests backward elimination on Quadratic regression for stability
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `QuadRegressionTest17` object tests the `QuadRegression` class using the Stability
 *  dataset.  It illustrates using the `Relation` class for reading the data
 *  from a .csv file "Stability_Data.csv".  Assumes no missing values.
 *  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 bar and R^2 cv vs. the instance index.
 *  > runMain scalation.analytics.QuadRegressionTest17
 */
object QuadRegressionTest17 extends App
{
    import scalation.columnar_db.Relation

   

    banner ("Stability relation")
    val auto_tab = Relation (BASE_DIR + "Stability_Data.csv", "Stability", null, -1)
    //auto_tab.show ()

    banner ("Stability (x, y) dataset")
    val (x, y) = auto_tab.toMatriDD (0 to 11, 12)
    println (s"x = $x")
    println (s"y = $y")

    banner ("Stability: QuadRegression")
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

} // QuadRegressionTest17 object


//Quadtest18 tests stepwise regression on Quadratic regression for stability
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `QuadRegressionTest18` object tests the `QuadRegression` class using the Stability
 *  dataset.  It illustrates using the `Relation` class for reading the data
 *  from a .csv file "Stability_Data.csv".  Assumes no missing values.
 *  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 bar and R^2 cv vs. the instance index.
 *  > runMain scalation.analytics.QuadRegressionTest18
 */
object QuadRegressionTest18 extends App
{
    import scalation.columnar_db.Relation

    banner ("Stability relation")
    val auto_tab = Relation (BASE_DIR + "Stability_Data.csv", "Stability", null, -1)
    //auto_tab.show ()

    banner ("Stability (x, y) dataset")
    val (x, y) = auto_tab.toMatriDD (0 to 11, 12)
    println (s"x = $x")
    println (s"y = $y")

    banner ("Stability: QuadRegression")
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

} // QuadRegressionTest18 object


//QuadXTest14 tests QuadX regression using forward selection on the stability set
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `QuadXRegressionTest14` object tests the `QuadXRegression` class using the Stability
 *  dataset.  It illustrates using the `Relation` class for reading the data
 *  from a .csv file "Stability_Data.csv".  Assumes no missing values.
 *  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 bar and R^2 cv vs. the instance index.
 *  > runMain scalation.analytics.QuadXRegressionTest14
 */
object QuadXRegressionTest14 extends App
{
    import scalation.columnar_db.Relation

    banner ("Stability relation")
    val auto_tab = Relation (BASE_DIR + "Stability_Data.csv", "Stability", null, -1)
    //auto_tab.show ()

    banner ("Stability (x, y) dataset")
    val (x, y) = auto_tab.toMatriDD (0 to 11, 12)
    println (s"x = $x")
    println (s"y = $y")

    banner ("Stability regression")
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

} // QuadXRegressionTest14 object


//QuadXTest15 tests QuadX regression using backward elimination on the stability set
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `QuadXRegressionTest15` object tests the `QuadXRegression` class using the Stability
 *  dataset.  It illustrates using the `Relation` class for reading the data
 *  from a .csv file "Stability_Data.csv".  Assumes no missing values.
 *  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 bar and R^2 cv vs. the instance index.
 *  > runMain scalation.analytics.QuadXRegressionTest15
 */
object QuadXRegressionTest15 extends App
{
    import scalation.columnar_db.Relation

    banner ("Stability relation")
    val auto_tab = Relation (BASE_DIR + "Stability_Data.csv", "Stability", null, -1)
    //auto_tab.show ()

    banner ("Stability (x, y) dataset")
    val (x, y) = auto_tab.toMatriDD (0 to 11, 12)
    println (s"x = $x")
    println (s"y = $y")

    banner ("Stability regression")
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

} // QuadXRegressionTest15 object


//QuadXTest16 tests QuadX regression using stepwise regression on the stability set
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `QuadXRegressionTest16` object tests the `QuadXRegression` class using the Stability
 *  dataset.  It illustrates using the `Relation` class for reading the data
 *  from a .csv file "Stability_Data.csv".  Assumes no missing values.
 *  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 bar and R^2 cv vs. the instance index.
 *  > runMain scalation.analytics.QuadXRegressionTest16
 */
object QuadXRegressionTest16 extends App
{
    import scalation.columnar_db.Relation

     banner ("Stability relation")
    val auto_tab = Relation (BASE_DIR + "Stability_Data.csv", "Stability", null, -1)
    //auto_tab.show ()

    banner ("Stability (x, y) dataset")
    val (x, y) = auto_tab.toMatriDD (0 to 11, 12)
    println (s"x = $x")
    println (s"y = $y")

    banner ("Stability regression")
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

} // QuadXRegressionTest16 object




//CubicTest14 tests cubic regression using forward selection on stability
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `CubicRegressionTest14` object tests the `CubicRegression` class using the Stability
 *  dataset.  It illustrates using the `Relation` class for reading the data
 *  from a .csv file "Stability_Data.csv".  Assumes no missing values.
 *  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 bar and R^2 cv vs. the instance index.
 *  > runMain scalation.analytics.CubicRegressionTest14
 */
object CubicRegressionTest14 extends App
{
    import scalation.columnar_db.Relation

    banner ("Stability relation")
    val auto_tab = Relation (BASE_DIR + "Stability_Data.csv", "Stability", null, -1)
    //auto_tab.show ()

    banner ("Stability (x, y) dataset")
    val (x, y) = auto_tab.toMatriDD (0 to 11, 12)

//  import ExampleStability.{x, y}

    println (s"x = $x")
    println (s"y = $y")

    banner ("Stability regression")
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

} // CubicRegressionTest14 object




//CubicTest15 tests cubic regression using backward elimination on stability
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `CubicRegressionTest15` object tests the `CubicRegression` class using the Stability
 *  dataset.  It illustrates using the `Relation` class for reading the data
 *  from a .csv file "Stability_Data.csv".  Assumes no missing values.
 *  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 bar and R^2 cv vs. the instance index.
 *  > runMain scalation.analytics.CubicRegressionTest15
 */
object CubicRegressionTest15 extends App
{
    import scalation.columnar_db.Relation

    banner ("Stability relation")
    val auto_tab = Relation (BASE_DIR + "Stability_Data.csv", "Stability", null, -1)
    //auto_tab.show ()

    banner ("Stability (x, y) dataset")
    val (x, y) = auto_tab.toMatriDD (0 to 11, 12)

//  import ExampleStability.{x, y}

    println (s"x = $x")
    println (s"y = $y")

    banner ("Stability regression")
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

} // CubicRegressionTest15 object

//CubicTest16 tests cubic regression using stepwise regression on stability
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `CubicRegressionTest16` object tests the `CubicRegression` class using the Stability
 *  dataset.  It illustrates using the `Relation` class for reading the data
 *  from a .csv file "Stability_Data.csv".  Assumes no missing values.
 *  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 bar and R^2 cv vs. the instance index.
 *  > runMain scalation.analytics.CubicRegressionTest16
 */
object CubicRegressionTest16 extends App
{
    import scalation.columnar_db.Relation

    banner ("Stability relation")
    val auto_tab = Relation (BASE_DIR + "Stability_Data.csv", "Stability", null, -1)
    //auto_tab.show ()

    banner ("Stability (x, y) dataset")
    val (x, y) = auto_tab.toMatriDD (0 to 11, 12)

//  import ExampleStability.{x, y}

    println (s"x = $x")
    println (s"y = $y")

    banner ("Stability regression")
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

} // CubicRegressionTest16 object



//cubicXtest14 tests forward selection on cubicX regression for stability
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `CubicXRegressionTest14` object tests the `CubicXRegression` class using the Stability
 *  dataset.  It illustrates using the `Relation` class for reading the data
 *  from a .csv file "Stability_Data.csv".  Assumes no missing values.
 *  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 bar and R^2 cv vs. the instance index.
 *  > runMain scalation.analytics.CubicXRegressionTest14
 */
object CubicXRegressionTest14 extends App
{
    import scalation.columnar_db.Relation

    banner ("Stability relation")
    val auto_tab = Relation (BASE_DIR + "Stability_Data.csv", "Stability", null, -1)
    //auto_tab.show ()

    banner ("Stability (x, y) dataset")
    val (x, y) = auto_tab.toMatriDD (0 to 11, 12)
    println (s"x = $x")
    println (s"y = $y")

    banner ("Stability regression")
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

} // CubicXRegressionTest14 object



//cubicXtest15 tests backward elimination on cubicX regression for stability
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `CubicXRegressionTest15` object tests the `CubicXRegression` class using the Stability
 *  dataset.  It illustrates using the `Relation` class for reading the data
 *  from a .csv file "Stability_Data.csv".  Assumes no missing values.
 *  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 bar and R^2 cv vs. the instance index.
 *  > runMain scalation.analytics.CubicXRegressionTest15
 */
object CubicXRegressionTest15 extends App
{
    import scalation.columnar_db.Relation

    banner ("Stability relation")
    val auto_tab = Relation (BASE_DIR + "Stability_Data.csv", "Stability", null, -1)
    //auto_tab.show ()

    banner ("Stability (x, y) dataset")
    val (x, y) = auto_tab.toMatriDD (0 to 11, 12)
    println (s"x = $x")
    println (s"y = $y")

    banner ("Stability regression")
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

} // CubicXRegressionTest15 object



//cubicXtest16 tests stepwise regression on cubicX regression for stability
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `CubicXRegressionTest16` object tests the `CubicXRegression` class using the Stability
 *  dataset.  It illustrates using the `Relation` class for reading the data
 *  from a .csv file "Stability_Data.csv".  Assumes no missing values.
 *  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 bar and R^2 cv vs. the instance index.
 *  > runMain scalation.analytics.CubicXRegressionTest16
 */
object CubicXRegressionTest16 extends App
{
    import scalation.columnar_db.Relation

    banner ("Stability relation")
    val auto_tab = Relation (BASE_DIR + "Stability_Data.csv", "Stability", null, -1)
    //auto_tab.show ()

    banner ("Stability (x, y) dataset")
    val (x, y) = auto_tab.toMatriDD (0 to 11, 12)
    println (s"x = $x")
    println (s"y = $y")

    banner ("Stability regression")
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

} // CubicXRegressionTest16 object




