

import org.apache.commons.csv.*
import org.tribuo.*
import org.tribuo.data.columnar.RowProcessor
import org.tribuo.data.columnar.processors.field.DoubleFieldProcessor
import org.tribuo.data.columnar.processors.response.FieldResponseProcessor
import org.tribuo.data.csv.CSVDataSource
import org.tribuo.regression.*
import org.tribuo.regression.evaluation.RegressionEvaluator
import org.tribuo.regression.xgboost.XGBoostRegressionTrainer
import java.io.FileReader
import java.io.FileWriter
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
import kotlin.math.sqrt
import java.time.*
import kotlin.math.*

/* ─────────────────── segédek ─────────────────── */

fun Double.toCsv()       = if (isNaN() || isInfinite()) "" else toString()
fun List<Double>.median(): Double =
    if (isEmpty()) Double.NaN else sorted().let { s ->
        val m = s.size / 2
        if (s.size % 2 == 0) (s[m - 1] + s[m]) / 2.0 else s[m]
    }
fun List<Double>.std(): Double =
    if (size < 2) 0.0 else {
        val m = average()
        sqrt(fold(0.0) { a, v -> a + (v - m) * (v - m) } / (size - 1))
    }

/* ───────── target-változat enum ───────── */

enum class TargetAgg(val suffix: String, val fn: (List<Double>) -> Double) {
    MEDIAN("median", List<Double>::median),
    MEAN  ("mean",   List<Double>::average)
}

/* ───────── nyers sor struktúra ───────── */

data class RawRow(val ts: Long, val feats: DoubleArray, val target: Double)

/* ╔═══════════════════════════╗
   ║ 1. AGGREGÁLÁS             ║
   ╚═══════════════════════════╝ */
fun aggregateCsv(
    input: String,
    outBase: String,
    tsCol: String,
    tgtCol: String,
    sep: Char = ','
) {
    /*── fejléc, feature-lista ──*/
    val header = FileReader(input).use { rdr ->
        CSVParser(rdr, CSVFormat.DEFAULT.builder().setDelimiter(sep)
            .setHeader().setSkipHeaderRecord(false).build()).headerNames
    }
    val featCols = header.filter { it != tsCol && it != tgtCol }
    require(featCols.isNotEmpty()) { "No feature columns." }

    /*── nyers beolvasás ──*/
    val raw = mutableListOf<RawRow>()
    val readFmt = CSVFormat.DEFAULT.builder().setDelimiter(sep)
        .setHeader().setSkipHeaderRecord(true).build()
    FileReader(input).use { rdr ->
        CSVParser(rdr, readFmt).use { prs ->
            for (rec in prs) {
                val ts   = rec.get(tsCol).trim().toDoubleOrNull()?.toLong() ?: continue
                val feats = DoubleArray(featCols.size) { i ->
                    rec.get(featCols[i]).trim().toDoubleOrNull() ?: Double.NaN
                }
                if (feats.any { it.isNaN() }) continue
                val tgt  = rec.get(tgtCol).trim().toDoubleOrNull() ?: continue
                raw += RawRow(ts, feats, tgt)
            }
        }
    }
    if (raw.isEmpty()) { println("No data in $input"); return }

    val grouped = raw.groupBy { it.ts }.toSortedMap()
    val buf = DoubleArray(raw.size)           // újrahasznosított

    /*── két target-CSV ──*/
    for (agg in TargetAgg.entries) {
        val out = "${outBase}_${agg.suffix}.csv"
        CSVPrinter(FileWriter(out),
            CSVFormat.DEFAULT.builder().setDelimiter(sep).build()).use { pr ->

            val head = listOf(tsCol) +
                    featCols.flatMap { listOf("${it}_mean","${it}_std","${it}_median") } +
                    listOf("count","cloudless_ratio","${tgtCol}_${agg.suffix}")
            pr.printRecord(head)

            for ((ts, rows) in grouped) {
                val n = rows.size
                val rec = mutableListOf<String>()

                rec += ts.toString()                  // timestamp

                /* feature-statisztikák */
                for (fi in featCols.indices) {
                    for (i in 0 until n) buf[i] = rows[i].feats[fi]
                    rec += listOf(
                        buf.take(n).average(),
                        buf.take(n).std(),
                        buf.take(n).toList().median()
                    ).map(Double::toCsv)
                }

                /* count + ratio */
                rec += n.toString()
                rec += (rows.count { it.target >= 0.95 }.toDouble() / n).toCsv()

                /* target */
                rec += agg.fn(rows.map { it.target }).toCsv()

                pr.printRecord(rec)
            }
        }
        println("✔ aggregated → $out")
    }
}

/* ╔═══════════════════════════╗
   ║ 2. XGBoost + preds        ║
   ╚═══════════════════════════╝ */
fun runXgb(
    trainCsv: String,
    testCsv : String,
    tsCol   : String,
    targetCol: String,
    sep: Char,
    tag: String
) {
    val header = FileReader(trainCsv).use { rdr ->
        CSVParser(rdr, CSVFormat.DEFAULT.builder().setDelimiter(sep)
            .setHeader().setSkipHeaderRecord(false).build()).headerNames
    }

    /* timestamp, target, ratio  kiszűrése */
    val featNames = header.filter {
        it != tsCol && it != targetCol && it != "cloudless_ratio"
    }

    val rowProc = RowProcessor.Builder<Regressor>()
        .setFieldProcessors(ArrayList(featNames.map { DoubleFieldProcessor(it) }))
        .build(FieldResponseProcessor(targetCol,"UNK",RegressionFactory()))

    val trainDS = MutableDataset(CSVDataSource(Path.of(trainCsv), rowProc, true))
    val testDS  = MutableDataset(CSVDataSource(Path.of(testCsv ), rowProc, true))

    println("▶ $tag | train=${trainDS.size()} test=${testDS.size()} feats=${featNames.size}")

    val model = XGBoostRegressionTrainer(1000).train(trainDS)
    //if (testD) { println("(skip eval – empty test)"); return }

    val eval = RegressionEvaluator().evaluate(model, testDS)
    println(eval)

    /* timestamp lista a teszt‐CSV-ből */
    val tsList = FileReader(testCsv).use { rdr ->
        CSVParser(rdr, CSVFormat.DEFAULT.builder().setDelimiter(sep)
            .setHeader().setSkipHeaderRecord(true).build()).map { it.get(tsCol) }
    }

    val out = Paths.get("${tag}_preds.csv")
    val lines = mutableListOf("timestamp,actual,predicted")
    eval.predictions.forEachIndexed { i, p ->
        lines += "${tsList[i]},${p.example.output.values[0]},${p.output.values[0]}"
    }
    Files.write(out, lines)
    println("✔ preds → $out")
}

/* ╔═══════════════════════════╗
   ║ 2.  XGBoost RAW           ║
   ╚═══════════════════════════╝ */
fun runXgbRaw(
    trainCsv : String,
    testCsv  : String,
    tsCol    : String,
    targetCol: String,
    sep      : Char,
    tag      : String
) {
    val enhTrain = trainCsv.removeSuffix(".csv") + "_enh.csv"
    val enhTest  = testCsv .removeSuffix(".csv") + "_enh.csv"
    enhanceRawCsv(trainCsv, enhTrain, tsCol, targetCol, sep)
    enhanceRawCsv(testCsv , enhTest , tsCol, targetCol, sep)

    /* 2️⃣  innentől ugyanaz, csak enhTrain/EnhTest-tel dolgozunk */
    val header = FileReader(enhTrain).use { rdr ->
        CSVParser(rdr, CSVFormat.DEFAULT.builder().setDelimiter(sep)
            .setHeader().setSkipHeaderRecord(false).build()).headerNames
    }
    /* kizárjuk tsCol + target */
    val featNames = header.filter { it != tsCol && it != targetCol }

    val rowProc = RowProcessor.Builder<Regressor>()
        .setFieldProcessors(ArrayList(featNames.map { DoubleFieldProcessor(it) }))
        .build(FieldResponseProcessor(targetCol,"UNK",RegressionFactory()))

    val trainDS = MutableDataset(CSVDataSource(Path.of(enhTrain), rowProc, true))
    val testDS  = MutableDataset(CSVDataSource(Path.of(enhTest ), rowProc, true))

    println("▶ RAW $tag | train=${trainDS.size()} test=${testDS.size()} feats=${featNames.size}")

    val params = mutableMapOf<String, Any>(

        "objective"        to "reg:squarederror",
        "eta"              to 0.05,
        "max_depth"        to 8,
        "subsample"        to 0.8,
        "colsample_bytree" to 0.8,
        "min_child_weight" to 1,
        "gamma"            to 0.0,
        "nthread"          to Runtime.getRuntime().availableProcessors()
    )

    val trainer = XGBoostRegressionTrainer(
        XGBoostRegressionTrainer.RegressionType.LINEAR,
        3000,
        params
    )



    val model = trainer.train(trainDS)
    val eval  = RegressionEvaluator().evaluate(model, testDS)
    println(eval)



    /*── írás ──*/
    val out = Paths.get("${tag}_preds.csv")
    val lines = mutableListOf("row,actual,predicted")
    eval.predictions.forEachIndexed { i, p ->
        lines += "${i+1},${p.example.output.values[0]},${p.output.values[0]}"
    }
    Files.write(out, lines)
    println("✔ RAW preds → $out")
}



/* ╔═══════════════════════════╗
   ║ 3. MAIN – két aggregált + │
   ║            egy nyers modell█
   ╚═══════════════════════════╝ */
fun main() {
    val trainSrc = "C:/Dev/smile/asd3honap_train.csv"
    val testSrc  = "C:/Dev/smile/asd3honap_test.csv"
    val tsCol    = "timestamp"
    val tgtCol   = "cloudTransparency"
    val sep      = ','

    /* 1️⃣  aggregált pipeline (mean & median) */
    aggregateCsv(trainSrc,"C:/Dev/smile/asd3honap_train_agg",tsCol,tgtCol,sep)
    aggregateCsv(testSrc ,"C:/Dev/smile/asd3honap_test_agg", tsCol,tgtCol,sep)
    for (agg in TargetAgg.entries) {
        val trainAgg = "C:/Dev/smile/asd3honap_train_agg_${agg.suffix}.csv"
        val testAgg  = "C:/Dev/smile/asd3honap_test_agg_${agg.suffix}.csv"
        val tgtAgg   = "${tgtCol}_${agg.suffix}"
        runXgb(trainAgg,testAgg,tsCol,tgtAgg,sep,"xgb_${agg.suffix}")
    }

    /* 2️⃣  nyers (aggregálatlan) pipeline */
    runXgbRaw(trainSrc, testSrc, tsCol, tgtCol, sep, "xgb_raw")
}

fun enhanceRawCsv(
    input : String,
    output: String,
    tsCol : String,
    tgtCol: String,
    sep   : Char = ','
) {
    val parserFmt = CSVFormat.DEFAULT.builder()
        .setDelimiter(sep).setHeader().setSkipHeaderRecord(true).build()
    val header = FileReader(input).use { rdr ->
        CSVParser(rdr, parserFmt.withSkipHeaderRecord(false)).headerNames
    }

    val numericCols = header.filter { it != tsCol && it != tgtCol }       // ts is non-numeric
    val hasHumCur   = "humidityPercentageCurrent" in header

    /*── build new header ──*/
    val newHeader = mutableListOf<String>()
    newHeader += header
    newHeader += "lag_$tgtCol"
    numericCols.forEach { newHeader += "lag_$it" }
    if (hasHumCur) newHeader += "diffHumidity"
    newHeader += listOf("hour_sin","hour_cos")

    /*── stream-read & write ──*/
    CSVPrinter(FileWriter(output),
        CSVFormat.DEFAULT.builder().setDelimiter(sep).build()).use { pr ->

        pr.printRecord(newHeader)
        var prevVals: Map<String,String>? = null

        FileReader(input).use { rdr ->
            CSVParser(rdr, parserFmt).use { prs ->
                for (rec in prs) {
                    val row = mutableListOf<String>()

                    /*── original cols ──*/
                    header.forEach { row += rec.get(it) }

                    /*── lag target ──*/
                    row += (prevVals?.get(tgtCol) ?: "")

                    /*── lag features ──*/
                    numericCols.forEach { col ->
                        row += (prevVals?.get(col) ?: "")
                    }

                    /*── diffHumidity ──*/
                    if (hasHumCur) {
                        val hTar = rec.get("humidityPercentageTarget").toDoubleOrNull()
                        val hCur = rec.get("humidityPercentageCurrent").toDoubleOrNull()
                        row += if (hTar!=null && hCur!=null) (hTar-hCur).toString() else ""
                    }

                    /*── hour_sin / cos ──*/
                    val ts = rec.get(tsCol).toDoubleOrNull()?.toLong()
                    if (ts!=null) {
                        val hour = Instant.ofEpochMilli(ts)
                            .atZone(ZoneOffset.UTC).hour.toDouble()
                        val angle = 2 * Math.PI * hour / 24.0
                        row += listOf(sin(angle), cos(angle)).map(Double::toString)
                    } else row += listOf("","")

                    pr.printRecord(row)
                    prevVals = header.associateWith { rec.get(it) }
                }
            }
        }
    }
    println("✔ enhanced → $output")
}
