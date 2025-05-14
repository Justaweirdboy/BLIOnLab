@file:OptIn(ExperimentalStdlibApi::class)

import ai.djl.Model
import ai.djl.ndarray.*
import ai.djl.ndarray.types.Shape
import ai.djl.nn.SequentialBlock
import ai.djl.nn.core.Linear
import ai.djl.nn.recurrent.LSTM
import ai.djl.training.DefaultTrainingConfig
import ai.djl.training.EasyTrain
import ai.djl.training.dataset.ArrayDataset
import ai.djl.training.dataset.Batch
import ai.djl.training.listener.TrainingListener
import ai.djl.training.loss.Loss
import ai.djl.training.optimizer.Adam
import ai.djl.training.util.MinMaxScaler
import ai.djl.translate.NoopTranslator
import org.apache.commons.csv.CSVFormat
import org.apache.commons.csv.CSVPrinter
import java.nio.file.Files
import java.nio.file.Paths
import kotlin.io.path.reader
import kotlin.math.sqrt

data class Row(val ts: Long, val feat: FloatArray, val label: Float)

/* ---------------------------------------------------------------------------------- *
 * 1. CSV beolvasás – NINCS aggregálás!
 * ---------------------------------------------------------------------------------- */
fun loadCsv(
    path: String,
    tsCol: String,
    featCols: List<String>,
    labelCol: String
): List<Row> =
    Files.newBufferedReader(Paths.get(path)).use { r ->
        CSVFormat.DEFAULT.builder().setHeader().setSkipHeaderRecord(true).build()
            .parse(r)
            .mapNotNull { rec ->
                val ts    = rec[tsCol].toDoubleOrNull()?.toLong() ?: return@mapNotNull null
                val feat  = featCols.map { rec[it].toFloat() }.toFloatArray()
                val label = rec[labelCol].toFloat()
                Row(ts, feat, label)
            }
            .sortedBy { it.ts }              // csak rendezés, nincs átlagolás
    }

/* ---------------------------------------------------------------------------------- *
 * 2. lista → NDArray
 * ---------------------------------------------------------------------------------- */
fun rowsToNd(rows: List<Row>, m: NDManager): Pair<NDArray, NDArray> =
    m.create(rows.flatMap { it.feat.asList() }.toFloatArray(),
        Shape(rows.size.toLong(), rows[0].feat.size.toLong())) to
            m.create(rows.map { it.label }.toFloatArray(), Shape(rows.size.toLong(), 1))

/* ---------------------------------------------------------------------------------- *
 * 3. skálázás + sliding window
 * ---------------------------------------------------------------------------------- */
fun windowed(
    x: NDArray,
    y: NDArray,
    win: Int,
    m: NDManager
): Triple<NDArray, NDArray, MinMaxScaler> {
    val sx = MinMaxScaler().apply { fit(x) }
    val sy = MinMaxScaler().apply { fit(y) }
    val xs = sx.transform(x)
    val ys = sy.transform(y)

    val n = xs.shape[0].toInt() // Use full size, no window-based reduction
    require(n >= win) { "túl rövid idősor ($n < $win)" }

    // Create windows for features
    val windows = Array(n - win + 1) { i -> xs.get("$i:${i + win},:") }
    val X = NDArrays.stack(NDList(*windows))
    // Use labels starting from index 0, no offset
    val Y = ys.get("0:${n - win + 1}")

    return Triple(X, Y, sy)
}

/* ---------------------------------------------------------------------------------- *
 * 4. főprogram
 * ---------------------------------------------------------------------------------- */
fun main() = NDManager.newBaseManager().use { m ->
    val feats = listOf(
        "cloudCoverAverageRgbTarget", "groupMeasurementToCloudCoverTransparencyCurrent",
        "humidityPercentageTarget",   "relativeAirmassCurrent",
        "relativeAirmassTarget",      "tempCelsiusTarget",
        "windSpeedKmPerHTarget"
    )

    // Load train and test CSVs separately
    val trainRows = loadCsv("C:/Dev/smile/asd3honap_train.csv", "timestamp", feats, "cloudTransparency")
    val testRows = loadCsv("C:/Dev/smile/asd3honap_test.csv", "timestamp", feats, "cloudTransparency")

    // Convert to NDArrays
    val (trainFx, trainFy) = rowsToNd(trainRows, m)
    val (testFx, testFy) = rowsToNd(testRows, m)

    val win = 1

    // Apply windowing and scaling
    val (trainX, trainY, sY) = windowed(trainFx, trainFy, win, m)
    val (testX, testY, _) = windowed(testFx, testFy, win, m)

    // Create datasets
    val trainSet = ArrayDataset.Builder()
        .setData(trainX)
        .optLabels(trainY)
        .setSampling(64, false)
        .build()

    val testSize = testX.shape[0].toInt()
    val testSet = ArrayDataset.Builder()
        .setData(testX)
        .optLabels(testY)
        .setSampling(testSize, false) // Single batch
        .build()

    /* ---- LSTM model ---- */
    val net = SequentialBlock()
        .add(
            LSTM.builder()
                .setStateSize(50)
                .setNumLayers(3)
                .optBatchFirst(true)
                .build()
        )
        .add { l -> NDList(l.head().get(":, -1, :")) }
        .add(Linear.builder().setUnits(1).build())


    Model.newInstance("cloud-transparency").use { model ->
        model.block = net

        val cfg = DefaultTrainingConfig(Loss.l2Loss())
            .optOptimizer(Adam.builder().build())
            .addEvaluator(Loss.l2Loss())
            .addTrainingListeners(*TrainingListener.Defaults.logging())

        model.newTrainer(cfg).use { tr ->
            tr.initialize(Shape(1, win.toLong(), feats.size.toLong()))
            EasyTrain.fit(tr, 10, trainSet, testSet)
            println(tr.getTrainingResult())
        }

        /* ---- Test prediction ---- */
        val testBatch: Batch = testSet.getData(m).first()
        val testData = testBatch.data.singletonOrThrow()
        val yTrueScaled = testBatch.labels.singletonOrThrow()

        model.newPredictor(NoopTranslator()).use { pred ->
            val yPredScaled = pred.predict(NDList(testData)).singletonOrThrow()

            val yPred = sY.inverseTransform(yPredScaled)
            val yTrue = sY.inverseTransform(yTrueScaled)

            val evaluator = Loss.l2Loss()

            val loss2 = evaluator.evaluate(NDList(yTrue), NDList(yPred))

            // Kiértékelés eredményei
            println("RMSE: ${sqrt(loss2.getFloat())}")

            /* ---- Write CSV ---- */
            Files.newBufferedWriter(Paths.get("predictions350lstm3layer.csv")).use { w ->
                CSVPrinter(
                    w,
                    CSVFormat.DEFAULT.builder()
                        .setHeader("row", "actual", "predicted")
                        .build()
                ).use { csv ->
                    for (i in 0 until testSize) {
                        csv.printRecord(
                            i,
                            yTrue.getFloat(i.toLong(), 0),
                            yPred.getFloat(i.toLong(), 0)
                        )
                    }
                }
            }
            println("✓ predictions.csv elkészült ($testSize sor)")
        }


        println(model.toString())
        model.save(Paths.get("export"), "cloud-transparency")
    }
}




