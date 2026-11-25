package com.example.ps5.core

import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.min

enum class Verdict(val label: String) {
    PASS("PASS"),
    RE_INSPECT("RE-INSPECT"),
    REJECT("REJECT"),
}

enum class Severity {
    LOW,
    MEDIUM,
    HIGH,
}

data class DecisionThresholds(
    val passScore: Float,
    val rejectScore: Float,
    val lowAreaCut: Float,
    val highAreaCut: Float,
    val confidenceScale: Float,
) {
    init {
        require(passScore.isFinite() && rejectScore.isFinite())
        require(passScore <= rejectScore)
        require(lowAreaCut.isFinite() && highAreaCut.isFinite())
        require(lowAreaCut >= 0f && lowAreaCut < highAreaCut)
        require(confidenceScale.isFinite() && confidenceScale > 0f)
    }
}

data class InspectionDecision(
    val verdict: Verdict,
    /** Null for PASS items: severity describes a detected defect only. */
    val severity: Severity?,
    val confidence: Float,
)

/** Defect types considered safety-relevant by the illustrative SOP (bump severity one level). */
val SAFETY_RELEVANT_TYPES = setOf(DefectType.CRACK_OR_SCRATCH, DefectType.CONTAMINATION_OR_COLOUR)

object DecisionCore {
    /**
     * Rules (plan 6.4 with FAQ severity labels):
     *  - severity from anomaly area share (calibrated tertile cuts), bumped one level for
     *    safety-relevant types (illustrative SOP), only for items at or above t_pass;
     *  - verdict: score < t_pass PASS; score >= t_reject or severity HIGH REJECT; else RE-INSPECT;
     *  - confidence: logistic of distance to the nearest threshold, scaled on calib.
     */
    fun decide(
        score: Float,
        areaFraction: Float,
        thresholds: DecisionThresholds,
        defectType: DefectType? = null,
    ): InspectionDecision {
        require(score.isFinite())
        require(areaFraction.isFinite() && areaFraction in 0f..1f)

        val flagged = score >= thresholds.passScore
        val severity = if (!flagged) {
            null
        } else {
            val base = when {
                areaFraction < thresholds.lowAreaCut -> Severity.LOW
                areaFraction < thresholds.highAreaCut -> Severity.MEDIUM
                else -> Severity.HIGH
            }
            if (defectType in SAFETY_RELEVANT_TYPES && base != Severity.HIGH) {
                Severity.entries[base.ordinal + 1]
            } else {
                base
            }
        }
        val verdict = when {
            !flagged -> Verdict.PASS
            score >= thresholds.rejectScore || severity == Severity.HIGH -> Verdict.REJECT
            else -> Verdict.RE_INSPECT
        }
        val thresholdDistance = min(
            abs(score - thresholds.passScore),
            abs(score - thresholds.rejectScore),
        )
        // Range 0.5 (on a threshold) .. 1.0 (far from both thresholds).
        val confidence = (1.0 / (1.0 + exp(-thresholdDistance / thresholds.confidenceScale)))
            .toFloat()

        return InspectionDecision(verdict, severity, confidence)
    }
}
