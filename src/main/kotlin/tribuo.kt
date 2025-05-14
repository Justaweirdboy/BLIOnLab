package org.example

import org.apache.commons.csv.CSVFormat
import org.apache.commons.csv.CSVParser
import org.apache.commons.csv.CSVPrinter
import org.tribuo.*
import org.tribuo.common.tree.AbstractCARTTrainer
import org.tribuo.common.tree.RandomForestTrainer
import org.tribuo.data.columnar.RowProcessor
import org.tribuo.data.columnar.processors.field.DoubleFieldProcessor
import org.tribuo.data.columnar.processors.response.FieldResponseProcessor
import org.tribuo.data.csv.CSVDataSource
import org.tribuo.ensemble.EnsembleCombiner
import org.tribuo.math.optimisers.AdaGrad
import org.tribuo.regression.*
import org.tribuo.regression.evaluation.RegressionEvaluation
import org.tribuo.regression.evaluation.RegressionEvaluator
import org.tribuo.regression.ensemble.AveragingCombiner
import org.tribuo.regression.rtree.CARTRegressionTrainer
import org.tribuo.regression.rtree.impurity.MeanSquaredError
import org.tribuo.regression.sgd.linear.LinearSGDTrainer
import org.tribuo.regression.sgd.objectives.SquaredLoss
import org.tribuo.regression.xgboost.XGBoostRegressionTrainer
import java.io.FileReader
import java.io.FileWriter
import java.nio.file.Files
import java.nio.file.Paths
import kotlin.math.sqrt

/* ─────────────────────────────  SEGÉDFÜGGVÉNYEK  ───────────────────────────── */

fun List<Double>.median(): Double {
    if (isEmpty()) return Double.NaN
    val s = sorted()
    val m = s.size / 2
    return if (s.size % 2 == 0) (s[m - 1] + s[m]) / 2.0 else s[m]
}

fun List<Double>.std(): Double {                 // minta szórás (n-1)
    if (size < 2) return Double.NaN
    val mean = average()
    val varSum = fold(0.0) { acc, v -> acc + (v - mean) * (v - mean) }
    return sqrt(varSum / (size - 1))
}

/* ────────────────────────  ADATKLASSZ A NYERS CSV-HEZ  ─────────────────────── */

data class RawRow(
    val timestamp: Long,
    val features: Map<String, Double>,
    val target: Double
)

/* ─────────────────────────────  AGGREGÁLÓ FÜGGVÉNY  ────────────────────────── */

fun aggregateCsv(
    inputCsvPath: String,
    outputCsvPath: String,
    timestampCol: String,
    featureCols: List<String>,
    targetCol: String,
    sep: Char = ','
) {
    val raw = mutableListOf<RawRow>()

    val readFmt = CSVFormat.DEFAULT.builder()
        .setDelimiter(sep)
        .setHeader()
        .setSkipHeaderRecord(true)
        .build()

    /* -------- 1.  Beolvasás -------- */
    FileReader(inputCsvPath).use { r ->
        CSVParser(r, readFmt).use { p ->
            for (rec in p) {
                try {
                    val ts = rec.get(timestampCol).trim().toDoubleOrNull()?.toLong()
                        ?: continue         // invalid timestamp – skip

                    val feats = mutableMapOf<String, Double>()
                    var ok = true
                    for (f in featureCols) {
                        val v = rec.get(f).trim().toDoubleOrNull()
                        if (v == null) { ok = false; break }
                        feats[f] = v
                    }
                    if (!ok) continue

                    val tgt = rec.get(targetCol).trim().toDoubleOrNull() ?: continue
                    raw += RawRow(ts, feats, tgt)

                } catch (_: Exception) { /* skip ill-formed row */ }
            }
        }
    }

    if (raw.isEmpty()) {
        println("Nincs érvényes sor – üres aggregált fájl.")
        FileWriter(outputCsvPath).use { w ->
            CSVPrinter(w, CSVFormat.DEFAULT.builder().setDelimiter(sep).build())
                .printRecord(featureCols.flatMap { listOf("${it}_mean", "${it}_std", "${it}_median") } +
                        listOf("count", targetCol))
        }
        return
    }

    /* -------- 2.  Csoportosítás időbélyegre -------- */
    val byTs = raw.groupBy { it.timestamp }
    val sortedTs = byTs.keys.sorted()

    /* -------- 3.  Fejléc felépítése -------- */
    val header = featureCols.flatMap { listOf("${it}_mean", "${it}_std", "${it}_median") } +
            listOf("count", targetCol)  // targetCol  ==  target median

    /* -------- 4.  Írás -------- */
    FileWriter(outputCsvPath).use { w ->
        CSVPrinter(w, CSVFormat.DEFAULT.builder().setDelimiter(sep).build()).use { pr ->
            pr.printRecord(header)

            for (ts in sortedTs) {
                val rows = byTs[ts]!!
                val n = rows.size

                val rowStats = mutableListOf<String>()

                for (f in featureCols) {
                    val vals = rows.map { it.features[f]!! }
                    val mean   = vals.average()
                    val std    = vals.std()
                    val median = vals.median()
                    rowStats += listOf(mean, std, median).map { it.takeIf { !it.isNaN() }?.toString() ?: "" }
                }

                rowStats += n.toString()                                   // count
                rowStats += rows.map { it.target }.median().toString()     // target median

                pr.printRecord(rowStats)
            }
        }
    }
    println("Aggregált fájl elkészült: $outputCsvPath")
}

/* ───────────────────────────────  MAIN ────────────────────────────────────── */

fun main() {
    /* --------  beállítás  -------- */
    val trainCsv   = "C:\\Dev\\smile\\asd3honap_train.csv"
    val testCsv    = "C:\\Dev\\smile\\asd3honap_test.csv"
    val target     = "cloudTransparency"
    val tsCol      = "timestamp"
    val sep        = ','

    val aggTrain   = "C:\\Dev\\smile\\asd3honap_train_agg.csv"
    val aggTest    = "C:\\Dev\\smile\\asd3honap_test_agg.csv"

    /* -------- 1.  fejléc olvasás az eredeti train-ből -------- */
    val origHeader = FileReader(trainCsv).use { r ->
        CSVParser(r, CSVFormat.DEFAULT.builder().setDelimiter(sep).setHeader().setSkipHeaderRecord(false).build())
            .headerNames
    }

    val featureCols = origHeader.filter { it != target && it != tsCol }
    require(featureCols.isNotEmpty()) { "Nincs aggregálható jellemző!" }

    /* -------- 2.  aggregálás train + test -------- */
    println("Train aggregálása…")
    aggregateCsv(trainCsv, aggTrain, tsCol, featureCols, target, sep)

    println("Test aggregálása…")
    aggregateCsv(testCsv, aggTest, tsCol, featureCols, target, sep)

    /* -------- 3.  Tribuo betöltés az aggregáltból -------- */
    val aggHeader = FileReader(aggTrain).use { r ->
        CSVParser(r, CSVFormat.DEFAULT.builder().setDelimiter(sep).setHeader().setSkipHeaderRecord(false).build())
            .headerNames
    }

    val tribuoFeatures = aggHeader.filter { it != target }      // target marad a medián
    val fieldProcs = tribuoFeatures.map { DoubleFieldProcessor(it) }
    val respProc   = FieldResponseProcessor(target, "UNK", RegressionFactory())

    val rowProcessor = RowProcessor.Builder<Regressor>()
        .setFieldProcessors(ArrayList(fieldProcs))
        .build(respProc)

    val trainDS = MutableDataset(CSVDataSource(Paths.get(aggTrain), rowProcessor, true))
    val testDS  = MutableDataset(CSVDataSource(Paths.get(aggTest),  rowProcessor, true))

    println("Aggregált train méret: ${trainDS.size()}, feature-szám: ${tribuoFeatures.size}")
    println("Aggregált test  méret: ${testDS.size()}")

    if (trainDS.size() == 0) return

    /* -------- 4.  Modell: XGBoost (példa) -------- */
    val trainer   = XGBoostRegressionTrainer(1000)
    val model     = trainer.train(trainDS)
    val evaluator = RegressionEvaluator()

    if (testDS.size() > 0) {
        val eval: RegressionEvaluation = evaluator.evaluate(model, testDS)
        println(eval)
        writePredictionsCSV("xgb-agg", eval)
    } else println("Üres test – nincs kiértékelés.")
}

/* ─────────────────────────  RANDOM-FOREST SEGÉD  ─────────────────────────── */

@JvmOverloads
fun randomForestTrainer(
    maxDepth: Int = 15,
    minChildWeight: Float = AbstractCARTTrainer.MIN_EXAMPLES.toFloat(),
    minImpurityDecrease: Float = 0f,
    fractionFeaturesInSplit: Float = 0.7f,
    numTrees: Int = 10,
    seed: Long = Trainer.DEFAULT_SEED,
    impurity: MeanSquaredError = MeanSquaredError(),
    combiner: EnsembleCombiner<Regressor> = AveragingCombiner()
): RandomForestTrainer<Regressor> {
    val base = CARTRegressionTrainer(
        maxDepth, minChildWeight, minImpurityDecrease,
        fractionFeaturesInSplit, false, impurity, seed
    )
    return RandomForestTrainer(base, combiner, numTrees, seed)
}

/* ─────────────────────────  PREDIKCIÓ CSV-BE ÍRÁSA  ───────────────────────── */

fun writePredictionsCSV(name: String, eval: RegressionEvaluation) {
    val out = Paths.get("$name-preds.csv")
    if (eval.predictions.isEmpty()) {
        Files.write(out, listOf("row,actual,predicted", "No predictions"))
        return
    }
    val lines = mutableListOf("row,actual,predicted")
    eval.predictions.forEachIndexed { i, p ->
        val truth = p.example.output.values.getOrElse(0) { Double.NaN }
        val pred  = p.output.values.getOrElse(0) { Double.NaN }
        lines += "${i + 1},$truth,$pred"
    }
    Files.write(out, lines)
    println("Predikciók kiírva ide: $out")
}
