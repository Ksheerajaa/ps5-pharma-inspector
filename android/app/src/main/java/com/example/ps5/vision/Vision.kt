package com.example.ps5.vision

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import com.example.ps5.core.FinderConstants
import com.example.ps5.core.FrameGate
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfDouble
import org.opencv.core.MatOfInt
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.nio.FloatBuffer
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.ceil
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sqrt

/** PatchCore-lite ONNX model: input 1x3x256x256 RGB in 0..1, outputs anomaly_map 1x1x256x256 and score [1]. */
class VisionModel(modelBytes: ByteArray, threads: Int = 4) : AutoCloseable {
    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession = env.createSession(
        modelBytes,
        OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(threads)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        },
    )
    val threads = threads

    data class Output(val map: FloatArray, val score: Float)

    /** [chw] must hold 3*256*256 floats, RGB planar, 0..1. */
    fun run(chw: FloatArray): Output {
        OnnxTensor.createTensor(env, FloatBuffer.wrap(chw), longArrayOf(1, 3, SIZE.toLong(), SIZE.toLong())).use { input ->
            session.run(mapOf("image" to input)).use { result ->
                val mapTensor = result.get("anomaly_map").get() as OnnxTensor
                val scoreTensor = result.get("score").get() as OnnxTensor
                val mapBuffer = mapTensor.floatBuffer
                val map = FloatArray(mapBuffer.remaining()).also { mapBuffer.get(it) }
                return Output(map, scoreTensor.floatBuffer.get(0))
            }
        }
    }

    override fun close() = session.close()

    companion object {
        const val SIZE = 256

        /** RGB 8-bit 256x256 Mat -> planar float CHW in 0..1 (matches torchvision ToTensor). */
        fun toChw(rgb256: Mat): FloatArray {
            require(rgb256.rows() == SIZE && rgb256.cols() == SIZE && rgb256.type() == CvType.CV_8UC3)
            val bytes = ByteArray(SIZE * SIZE * 3)
            rgb256.get(0, 0, bytes)
            val plane = SIZE * SIZE
            val out = FloatArray(plane * 3)
            for (i in 0 until plane) {
                out[i] = (bytes[i * 3].toInt() and 0xFF) / 255f
                out[plane + i] = (bytes[i * 3 + 1].toInt() and 0xFF) / 255f
                out[2 * plane + i] = (bytes[i * 3 + 2].toInt() and 0xFF) / 255f
            }
            return out
        }
    }
}

enum class FrameIssue(val reason: String) { DARK("dark"), BRIGHT("bright"), GLARE("glare"), BLUR("blur") }

data class GateStats(val p1: Double, val p99: Double, val saturatedFraction: Double)

object QualityGate {
    fun stats(gray: Mat): GateStats {
        val hist = IntArray(256)
        val bytes = ByteArray(gray.rows() * gray.cols())
        gray.get(0, 0, bytes)
        for (b in bytes) hist[b.toInt() and 0xFF]++
        val n = bytes.size.toDouble()
        fun percentile(p: Double): Double {
            var acc = 0
            val target = p * n
            for (v in 0..255) {
                acc += hist[v]
                if (acc >= target) return v.toDouble()
            }
            return 255.0
        }
        var sat = 0
        for (v in FrameGate.SATURATED_LUMA.toInt()..255) sat += hist[v]
        return GateStats(percentile(0.01), percentile(0.99), sat / n)
    }

    fun check(s: GateStats): FrameIssue? = when {
        s.p99 < FrameGate.DARK_P99_LUMA -> FrameIssue.DARK
        s.p1 > FrameGate.BRIGHT_P1_LUMA -> FrameIssue.BRIGHT
        s.saturatedFraction > FrameGate.GLARE_SATURATED_FRACTION -> FrameIssue.GLARE
        else -> null
    }

    fun laplacianVariance(gray: Mat): Double {
        val lap = Mat()
        Imgproc.Laplacian(gray, lap, CvType.CV_64F)
        val mean = MatOfDouble(); val std = MatOfDouble()
        Core.meanStdDev(lap, mean, std)
        lap.release()
        val sd = std.toArray()[0]
        return sd * sd
    }
}

/** A reference item (addendum 2.2 setup step). areaFraction is relative to the finder work image area. */
data class ReferenceItem(
    val id: String,
    val areaFraction: Double,
    val aspect: Double,
    val meanRgb: DoubleArray?,
)

enum class BlobStatus { OK, CLUSTER, PARTIAL, UNKNOWN_OBJECT, TOO_MANY }

data class Blob(
    val status: BlobStatus,
    val reason: String?,
    /** Box in original-frame pixels. */
    val box: Rect,
    val centerX: Double,
    val centerY: Double,
    /** Long side in original-frame pixels. */
    val longSide: Double,
    /** Long-axis angle in degrees (image coordinates, y down). */
    val angleDeg: Double,
    val areaFraction: Double,
    val aspect: Double,
    val solidity: Double,
    val meanRgb: DoubleArray,
)

object ItemFinder {
    /** Default reference derived from MVTec pill train_good (06_calibrate.py, aspect from minAreaRect). */
    fun mvtecDefault(areaFraction: Double) = ReferenceItem("mvtec_pill_default", areaFraction, 1.46, null)

    fun find(rgb: Mat, reference: ReferenceItem?): List<Blob> {
        val scale = min(1.0, FinderConstants.WORK_MAX_SIDE.toDouble() / max(rgb.cols(), rgb.rows()))
        val work = Mat()
        Imgproc.resize(rgb, work, Size(rgb.cols() * scale, rgb.rows() * scale), 0.0, 0.0, Imgproc.INTER_AREA)
        val gray = Mat()
        Imgproc.cvtColor(work, gray, Imgproc.COLOR_RGB2GRAY)
        Imgproc.GaussianBlur(gray, gray, Size(5.0, 5.0), 0.0)
        val binary = Mat()
        Imgproc.threshold(gray, binary, 0.0, 255.0, Imgproc.THRESH_BINARY + Imgproc.THRESH_OTSU)
        if (borderMajority(binary)) Core.bitwise_not(binary, binary)
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(5.0, 5.0))
        Imgproc.morphologyEx(binary, binary, Imgproc.MORPH_OPEN, kernel)
        Imgproc.morphologyEx(binary, binary, Imgproc.MORPH_CLOSE, kernel)

        val labels = Mat(); val stats = Mat(); val centroids = Mat()
        val n = Imgproc.connectedComponentsWithStats(binary, labels, stats, centroids, 8, CvType.CV_32S)
        val workArea = work.rows().toDouble() * work.cols()
        val minArea = if (reference != null) reference.areaFraction * workArea * FinderConstants.NOISE_FRACTION_OF_REF else workArea * 0.002
        val blobs = mutableListOf<Blob>()
        for (i in 1 until n) {
            val area = stats.get(i, Imgproc.CC_STAT_AREA)[0]
            if (area < minArea) continue
            val x = stats.get(i, Imgproc.CC_STAT_LEFT)[0].toInt()
            val y = stats.get(i, Imgproc.CC_STAT_TOP)[0].toInt()
            val w = stats.get(i, Imgproc.CC_STAT_WIDTH)[0].toInt()
            val h = stats.get(i, Imgproc.CC_STAT_HEIGHT)[0].toInt()
            val roi = Rect(x, y, w, h)
            val mask = Mat()
            Core.compare(labels.submat(roi), Scalar(i.toDouble()), mask, Core.CMP_EQ)
            val contours = mutableListOf<MatOfPoint>()
            Imgproc.findContours(mask.clone(), contours, Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE)
            val contour = contours.maxByOrNull { Imgproc.contourArea(it) } ?: continue
            val hullIdx = MatOfInt()
            Imgproc.convexHull(contour, hullIdx)
            val pts = contour.toArray()
            val hull = MatOfPoint(*hullIdx.toArray().map { pts[it] }.toTypedArray())
            val solidity = Imgproc.contourArea(contour) / max(Imgproc.contourArea(hull), 1.0)
            val rect = Imgproc.minAreaRect(MatOfPoint2f(*pts))
            val longSide = max(rect.size.width, rect.size.height)
            val aspect = longSide / max(min(rect.size.width, rect.size.height), 1.0)
            val m = Imgproc.moments(contour)
            val angle = Math.toDegrees(0.5 * atan2(2 * m.mu11, m.mu20 - m.mu02))
            val meanRgb = Core.mean(work.submat(roi), mask).`val`.copyOf(3)
            val cx = centroids.get(i, 0)[0]; val cy = centroids.get(i, 1)[0]
            val touches = x <= 1 || y <= 1 || x + w >= work.cols() - 1 || y + h >= work.rows() - 1
            val areaFraction = area / workArea

            var status = BlobStatus.OK
            var reason: String? = null
            if (touches) {
                status = BlobStatus.PARTIAL; reason = "touches frame edge"
            } else if (reference != null) {
                val ratio = areaFraction / reference.areaFraction
                val aspectDev = abs(aspect / reference.aspect - 1)
                val colourDist = reference.meanRgb?.let { ref ->
                    sqrt((0..2).sumOf { (meanRgb[it] - ref[it]) * (meanRgb[it] - ref[it]) })
                } ?: 0.0
                when {
                    // Colour first: a different product or foreign object is UNKNOWN even if large.
                    colourDist > FinderConstants.MAX_COLOUR_DISTANCE ->
                        { status = BlobStatus.UNKNOWN_OBJECT; reason = "colour distance %.0f".format(colourDist) }
                    ratio > 1 + FinderConstants.AREA_BAND ->
                        { status = BlobStatus.CLUSTER; reason = "area %.2fx reference".format(ratio) }
                    solidity < FinderConstants.MIN_SOLIDITY ->
                        { status = BlobStatus.CLUSTER; reason = "solidity %.2f".format(solidity) }
                    ratio < 1 - FinderConstants.AREA_BAND ->
                        { status = BlobStatus.UNKNOWN_OBJECT; reason = "area %.2fx reference".format(ratio) }
                    aspectDev > FinderConstants.ASPECT_BAND ->
                        { status = BlobStatus.UNKNOWN_OBJECT; reason = "aspect %.2f vs %.2f".format(aspect, reference.aspect) }
                }
            } else if (solidity < FinderConstants.MIN_SOLIDITY) {
                status = BlobStatus.CLUSTER; reason = "solidity %.2f".format(solidity)
            }
            blobs += Blob(
                status, reason,
                Rect((x / scale).roundToInt(), (y / scale).roundToInt(), (w / scale).roundToInt(), (h / scale).roundToInt()),
                cx / scale, cy / scale, longSide / scale, angle, areaFraction, aspect, solidity, meanRgb,
            )
            mask.release()
        }
        work.release(); gray.release(); binary.release(); labels.release(); stats.release(); centroids.release()
        // Reading order: rows top to bottom, then left to right.
        val rowHeight = blobs.maxOfOrNull { it.box.height }?.toDouble() ?: 1.0
        val ordered = blobs.sortedWith(compareBy({ (it.centerY / rowHeight).toInt() }, { it.centerX }))
        var okCount = 0
        return ordered.map {
            if (it.status == BlobStatus.OK && ++okCount > FinderConstants.MAX_ITEMS) {
                it.copy(status = BlobStatus.TOO_MANY, reason = "more than ${FinderConstants.MAX_ITEMS} items")
            } else it
        }
    }

    private fun borderMajority(binary: Mat): Boolean {
        val r = binary.rows(); val c = binary.cols()
        val top = Core.countNonZero(binary.row(0)); val bottom = Core.countNonZero(binary.row(r - 1))
        val left = Core.countNonZero(binary.col(0)); val right = Core.countNonZero(binary.col(c - 1))
        return (top + bottom + left + right) > (2 * (r + c)) / 2
    }

    /**
     * Rotate the item's long axis to horizontal (MVTec pill orientation) and crop a square so the item's
     * long side fills [targetSideFraction] of the 256x256 crop. Out-of-frame area is filled with [bgRgb].
     */
    fun normalisedCrop(rgb: Mat, blob: Blob, targetSideFraction: Float, bgRgb: IntArray, rotate: Boolean = true): Mat {
        val side = blob.longSide / targetSideFraction
        val half = ceil(side * 0.75).toInt() // 1.5x side to keep corners after rotation
        val bg = Scalar(bgRgb[0].toDouble(), bgRgb[1].toDouble(), bgRgb[2].toDouble())
        val cx = blob.centerX.roundToInt(); val cy = blob.centerY.roundToInt()
        val padded = Mat()
        Core.copyMakeBorder(rgb, padded, half, half, half, half, Core.BORDER_CONSTANT, bg)
        val roi = Mat(padded, Rect(cx, cy, 2 * half, 2 * half))
        val rotated = Mat()
        val angle = if (rotate && abs(blob.angleDeg) > 2.0) blob.angleDeg else 0.0
        if (angle != 0.0) {
            val m = Imgproc.getRotationMatrix2D(Point(half.toDouble(), half.toDouble()), angle, 1.0)
            Imgproc.warpAffine(roi, rotated, m, roi.size(), Imgproc.INTER_LINEAR, Core.BORDER_CONSTANT, bg)
        } else {
            roi.copyTo(rotated)
        }
        val s = side.roundToInt().coerceAtLeast(8)
        val start = half - s / 2
        val square = Mat(rotated, Rect(start, start, s, s))
        val out = Mat()
        Imgproc.resize(square, out, Size(256.0, 256.0), 0.0, 0.0, Imgproc.INTER_AREA)
        padded.release(); rotated.release()
        return out
    }

    /** Whole-image resize used for MVTec single-item replay (identical to training preprocessing). */
    fun wholeImage256(rgb: Mat): Mat {
        val out = Mat()
        Imgproc.resize(rgb, out, Size(256.0, 256.0), 0.0, 0.0, Imgproc.INTER_AREA)
        return out
    }
}
