package com.example.ps5.pipeline

import android.graphics.Bitmap
import android.os.SystemClock
import com.example.ps5.core.AnomalyAnalysis
import com.example.ps5.core.AnomalyFeatures
import com.example.ps5.core.CalibratedConfig
import com.example.ps5.core.DecisionCore
import com.example.ps5.core.DefectTyper
import com.example.ps5.core.InspectionDecision
import com.example.ps5.vision.Blob
import com.example.ps5.vision.BlobStatus
import com.example.ps5.vision.FrameIssue
import com.example.ps5.vision.GateStats
import com.example.ps5.vision.ItemFinder
import com.example.ps5.vision.QualityGate
import com.example.ps5.vision.ReferenceItem
import com.example.ps5.vision.VisionModel
import org.opencv.android.Utils
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.sqrt

enum class FrameStatus { OK, NO_ITEMS, BAD_IMAGE }

data class ItemResult(
    val idx: Int,
    val blob: Blob,
    val score: Float? = null,
    val decision: InspectionDecision? = null,
    val features: AnomalyFeatures? = null,
    val defectType: String? = null,
    val typeBasis: String? = null,
    val crop: Bitmap? = null,
    val heat: Bitmap? = null,
    val preprocessMs: Long? = null,
    val visionMs: Long? = null,
)

data class FrameResult(
    val frameId: String,
    val status: FrameStatus,
    val reason: String?,
    val gate: GateStats?,
    val items: List<ItemResult>,
    val frameMs: Long,
    val width: Int,
    val height: Int,
)

class InspectionPipeline(
    private val model: VisionModel,
    val config: CalibratedConfig,
) {
    fun inspect(bitmap: Bitmap, frameId: String, reference: ReferenceItem?, keepImages: Boolean = true): FrameResult {
        val t0 = SystemClock.elapsedRealtime()
        val rgba = Mat()
        Utils.bitmapToMat(bitmap, rgba)
        val rgb = Mat()
        Imgproc.cvtColor(rgba, rgb, Imgproc.COLOR_RGBA2RGB)
        rgba.release()
        try {
            // 1. Frame gate on a 256-px-scale grey image (same domain as the gate statistics).
            val small = Mat()
            val s = 256.0 / maxOf(rgb.cols(), rgb.rows())
            Imgproc.resize(rgb, small, Size(rgb.cols() * s, rgb.rows() * s), 0.0, 0.0, Imgproc.INTER_AREA)
            Imgproc.cvtColor(small, small, Imgproc.COLOR_RGB2GRAY)
            val stats = QualityGate.stats(small)
            small.release()
            // Bright and glare are judged on the whole frame. "Dark" is judged only when items are present:
            // an empty tray on the dark fixture is legitimately dark (it becomes NO_ITEMS).
            val issue = QualityGate.check(stats)
            if (issue != null && issue != FrameIssue.DARK) {
                return FrameResult(frameId, FrameStatus.BAD_IMAGE, issue.reason, stats, emptyList(), SystemClock.elapsedRealtime() - t0, rgb.cols(), rgb.rows())
            }
            // 2. Item finder.
            val blobs = ItemFinder.find(rgb, reference)
            if (blobs.isEmpty()) {
                return FrameResult(frameId, FrameStatus.NO_ITEMS, "place items", stats, emptyList(), SystemClock.elapsedRealtime() - t0, rgb.cols(), rgb.rows())
            }
            if (issue == FrameIssue.DARK) {
                return FrameResult(frameId, FrameStatus.BAD_IMAGE, issue.reason, stats, emptyList(), SystemClock.elapsedRealtime() - t0, rgb.cols(), rgb.rows())
            }
            // 3. Crops of OK items + per-crop blur gate.
            val crops = mutableMapOf<Int, Pair<Mat, Long>>()
            blobs.forEachIndexed { i, blob ->
                if (blob.status != BlobStatus.OK) return@forEachIndexed
                val tp = SystemClock.elapsedRealtime()
                val crop = ItemFinder.normalisedCrop(rgb, blob, config.targetItemSideFraction, config.backgroundRgb)
                crops[i] = crop to (SystemClock.elapsedRealtime() - tp)
            }
            for ((_, c) in crops) {
                val g = Mat()
                Imgproc.cvtColor(c.first, g, Imgproc.COLOR_RGB2GRAY)
                val lv = QualityGate.laplacianVariance(g)
                g.release()
                if (lv < config.blurThreshold) {
                    crops.values.forEach { it.first.release() }
                    return FrameResult(frameId, FrameStatus.BAD_IMAGE, FrameIssue.BLUR.reason, stats, emptyList(), SystemClock.elapsedRealtime() - t0, rgb.cols(), rgb.rows())
                }
            }
            // 4. Vision + decision per OK item, sequential, batch size 1.
            val items = blobs.mapIndexed { i, blob ->
                val c = crops[i] ?: return@mapIndexed ItemResult(i + 1, blob)
                val (crop, prepMs) = c
                val r = inspectCrop(crop, keepImages).copy(idx = i + 1, blob = blob, preprocessMs = prepMs)
                crop.release()
                r
            }
            return FrameResult(frameId, FrameStatus.OK, null, stats, items, SystemClock.elapsedRealtime() - t0, rgb.cols(), rgb.rows())
        } finally {
            rgb.release()
        }
    }

    /** Vision + decision on one 256x256 RGB crop. Public for the golden-image and latency benchmarks. */
    fun inspectCrop(crop: Mat, keepImages: Boolean): ItemResult {
        val tp = SystemClock.elapsedRealtime()
        val chw = VisionModel.toChw(crop)
        val tv = SystemClock.elapsedRealtime()
        val out = model.run(chw)
        val visionMs = SystemClock.elapsedRealtime() - tv
        val features = AnomalyAnalysis.analyse(out.map, VisionModel.SIZE, config.tPix)
        val flagged = out.score >= config.thresholds.passScore
        var typeLabel: String? = null
        var basis: String? = null
        var typed: com.example.ps5.core.DefectType? = null
        if (flagged) {
            val t = DefectTyper.classify(features, colourDeltaE(crop, features), config.thresholds.highAreaCut)
            typed = t.type; typeLabel = t.type.label; basis = "indicative rule: ${t.basis}"
        }
        val decision = DecisionCore.decide(out.score, features.areaFraction, config.thresholds, typed)
        val cropBitmap = if (keepImages) toBitmap(crop) else null
        val heat = if (keepImages) heatmap(out.map) else null
        return ItemResult(
            idx = 0, blob = DUMMY_BLOB, score = out.score, decision = decision, features = features,
            defectType = typeLabel ?: "none", typeBasis = basis, crop = cropBitmap, heat = heat,
            preprocessMs = tv - tp, visionMs = visionMs,
        )
    }

    /** Mean CIE Lab distance between the anomaly blob and the median item colour in the crop. */
    private fun colourDeltaE(crop: Mat, f: AnomalyFeatures): Double {
        val mask = f.largestMask ?: return 0.0
        val lab = Mat()
        val cropF = Mat()
        crop.convertTo(cropF, CvType.CV_32FC3, 1.0 / 255)
        Imgproc.cvtColor(cropF, lab, Imgproc.COLOR_RGB2Lab)
        val px = FloatArray(256 * 256 * 3)
        lab.get(0, 0, px)
        lab.release(); cropF.release()
        // Item pixels: L above the midpoint between background and item (dark background fixture).
        val item = (0 until 256 * 256).filter { px[it * 3] > 35f && !mask[it] }
        if (item.isEmpty()) return 0.0
        fun median(ch: Int) = item.map { px[it * 3 + ch] }.sorted()[item.size / 2]
        val ref = floatArrayOf(median(0), median(1), median(2))
        var sum = 0.0; var n = 0
        for (i in mask.indices) {
            if (!mask[i]) continue
            val dl = px[i * 3] - ref[0]; val da = px[i * 3 + 1] - ref[1]; val db = px[i * 3 + 2] - ref[2]
            sum += sqrt((dl * dl + da * da + db * db).toDouble()); n++
        }
        return if (n == 0) 0.0 else sum / n
    }

    private fun toBitmap(rgb: Mat): Bitmap {
        val bmp = Bitmap.createBitmap(rgb.cols(), rgb.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(rgb, bmp)
        return bmp
    }

    private fun heatmap(map: FloatArray): Bitmap {
        val lo = config.thresholds.passScore * 0.5f
        val hi = config.thresholds.rejectScore * 1.2f
        val colors = IntArray(map.size) {
            val t = ((map[it] - lo) / (hi - lo)).coerceIn(0f, 1f)
            val a = (t * 200).toInt()
            (a shl 24) or (255 shl 16) or (((1 - t) * 200).toInt() shl 8)
        }
        return Bitmap.createBitmap(colors, 256, 256, Bitmap.Config.ARGB_8888)
    }

    companion object {
        private val DUMMY_BLOB = Blob(BlobStatus.OK, null, org.opencv.core.Rect(0, 0, 256, 256), 128.0, 128.0, 256.0, 0.0, 1.0, 1.0, 1.0, doubleArrayOf(0.0, 0.0, 0.0))
    }
}
