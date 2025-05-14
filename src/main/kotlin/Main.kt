import org.apache.commons.csv.CSVFormat
import org.apache.commons.csv.CSVPrinter
import smile.data.DataFrame
import smile.data.formula.Formula
import smile.io.Read
import smile.validation.metric.MSE
import smile.regression.*
import smile.timeseries.ARMA
import java.io.FileWriter
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.sqrt

// --------------------------------------------------------------------------
// 1) Segédfüggvények
// --------------------------------------------------------------------------

fun diff(series: DoubleArray, d: Int): DoubleArray {
    var s = series
    repeat(d) { s = DoubleArray(s.size - 1) { i -> s[i + 1] - s[i] } }
    return s
}

fun undiff(last: Double, diffForecast: DoubleArray): DoubleArray {
    val out = DoubleArray(diffForecast.size)
    var prev = last
    for (i in diffForecast.indices) {
        prev += diffForecast[i]
        out[i] = prev
    }
    return out
}

fun selectARMAParams(
    diffSeries: DoubleArray,
    maxP: Int = 5,
    maxQ: Int = 5
): Pair<Int, Int> {
    println("Auto-ARIMA: grid-search p=0..$maxP, q=0..$maxQ")
    var bestAic = Double.POSITIVE_INFINITY
    var bestP = 0
    var bestQ = 0
    val n = diffSeries.size.toDouble()
    for (p in 0..maxP) {
        for (q in 0..maxQ) {
            try {
                val m = ARMA.fit(diffSeries, p, q)
                val k = (p + q + 1).toDouble()
                val aic = n * ln(m.variance()) + 2 * k
                if (aic < bestAic) {
                    bestAic = aic
                    bestP = p
                    bestQ = q
                }
                println("  ARMA($p,$q) → AIC=$aic")
            } catch (_: Exception) {
                // invalid p/q vagy túl rövid sor → kihagyjuk
            }
        }
    }
    println("Auto-ARIMA: választott (p,q)=($bestP,$bestQ), AIC=$bestAic")
    return bestP to bestQ
}

fun rollingForecastARIMAWindow(
    train: DoubleArray,
    test: DoubleArray,
    d: Int,
    p: Int,
    q: Int,
    window: Int
): DoubleArray {
    println("Auto-ARIMA Windowed: rolling (d=$d,p=$p,q=$q,window=$window) for ${test.size} steps")
    val preds = DoubleArray(test.size)
    val minLen = p + q + 20 + max(p, q) // ARMA.fit követelménye

    // A tanító adatok indexeinek kezelése az ablak csúsztatásához
    for (i in test.indices) {
        // Az ablak kezdő és végső indexe a tanító adatokon belül
        val startIdx = max(0, train.size - window - i)
        val endIdx = train.size - i
        val slice = if (endIdx - startIdx >= minLen) {
            train.sliceArray(startIdx until endIdx)
        } else {
            train.sliceArray(max(0, train.size - minLen) until train.size)
        }

        val diffHist = diff(slice, d)
        var nextPred = train[train.size - i - 1] // persistence fallback

        if (diffHist.size >= minLen) {
            try {
                val arma = ARMA.fit(diffHist, p, q)
                val dp = arma.forecast(1)[0]
                nextPred = train[train.size - i - 1] + dp
            } catch (e: Exception) {
                println("  [WARN] step ${i+1}: ARMA.fit/forecast failed, using persistence")
            }
        } else {
            println("  [INFO] step ${i+1}: slice too short (${diffHist.size} < $minLen), persistence")
        }

        preds[i] = nextPred

        if ((i+1) % 10 == 0 || i == test.lastIndex) {
            println("  step ${i+1}/${test.size}: pred=$nextPred, actual=${test[i]}")
        }
    }

    println("Auto-ARIMA Windowed: done")
    return preds
}

class ARIMARollingWindowModel(
    val trainSeries: DoubleArray,
    val testSeries: DoubleArray,
    val d: Int,
    val p: Int,
    val q: Int,
    val window: Int
) {
    fun forecast(): DoubleArray =
        rollingForecastARIMAWindow(trainSeries, testSeries, d, p, q, window)
}

// --------------------------------------------------------------------------
// 2) main – pipeline
// --------------------------------------------------------------------------

fun main() {
    val fmt = CSVFormat.DEFAULT.builder()
        .setHeader()
        .setSkipHeaderRecord(true)
        .build()

    val trainDF = Read.csv("C:\\Dev\\smile\\asd3honap_train.csv", fmt)
    val testDF  = Read.csv("C:\\Dev\\smile\\asd3honap_test.csv", fmt)
    val target  = "cloudTransparency"
    val formula = Formula.lhs(target)

    val models = mapOf<String, () -> Any>(
        "RandomForest" to {
            println("Training RandomForest…")
            randomForest(formula, trainDF,
                ntrees=200, mtry=0, maxDepth=10, nodeSize=4, subsample=0.8)
        },
        "GradientBoosting" to {
            println("Training GradientBoosting…")
            gbm(formula, trainDF,
                ntrees=200, maxDepth=6, nodeSize=4, shrinkage=0.1, subsample=0.8)
        },
        "Ridge" to {
            println("Training Ridge Regression…")
            ridge(formula, trainDF, lambda=1.0)
        },
        "Auto-ARIMA-Windowed" to {
            println("Preparing Auto-ARIMA Windowed model with W=1000…")
            val series    = trainDF.column(target).toDoubleArray()
            val testSeries= testDF.column(target).toDoubleArray()
            val d = 1
            val (p, q) = selectARMAParams(diff(series, d), maxP=5, maxQ=5)
            ARIMARollingWindowModel(series, testSeries, d, p, q, window=1000)
        }
    )

    models.forEach { (name, trainFn) ->
        println("\n===== Model: $name =====")
        val model = trainFn()
        val preds = when (model) {
            is RandomForest          ->
                DoubleArray(testDF.nrow()) { i -> model.predict(testDF[i]) }
            is GradientTreeBoost     ->
                DoubleArray(testDF.nrow()) { i -> model.predict(testDF[i]) }
            is LinearModel           ->
                DoubleArray(testDF.nrow()) { i -> model.predict(testDF[i]) }
            is ARIMARollingWindowModel ->
                model.forecast()
            else -> error("Unknown model type: ${model::class}")
        }

        val actual = testDF.column(target).toDoubleArray()
        val mse = MSE.of(actual, preds)
        val rmse= sqrt(mse)
        println("$name → MSE=$mse, RMSE=$rmse")

        val out = "C:\\Dev\\smile\\predictions_${name.replace(" ", "_")}.csv"
        FileWriter(out).use { w ->
            val p = CSVPrinter(w, CSVFormat.DEFAULT.builder()
                .setHeader("row","actual","predicted").build())
            actual.indices.forEach { i -> p.printRecord(i, actual[i], preds[i]) }
            p.flush()
        }
        println("$name saved to $out")
    }

    println("\nAll done.")
}