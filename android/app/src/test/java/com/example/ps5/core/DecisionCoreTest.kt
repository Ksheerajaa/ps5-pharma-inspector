package com.example.ps5.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class DecisionCoreTest {
    private val thresholds = DecisionThresholds(
        passScore = 0.2f,
        rejectScore = 0.8f,
        lowAreaCut = 0.01f,
        highAreaCut = 0.05f,
        confidenceScale = 0.1f,
    )

    @Test
    fun scoreBelowPassCutReturnsPassWithoutSeverity() {
        val result = DecisionCore.decide(0.1f, 0.2f, thresholds)
        assertEquals(Verdict.PASS, result.verdict)
        assertNull(result.severity)
    }

    @Test
    fun scoreBetweenCutsReturnsReinspect() {
        val result = DecisionCore.decide(0.5f, 0.02f, thresholds)
        assertEquals(Verdict.RE_INSPECT, result.verdict)
        assertEquals(Severity.MEDIUM, result.severity)
    }

    @Test
    fun scoreAboveRejectCutReturnsReject() {
        val result = DecisionCore.decide(0.9f, 0.001f, thresholds)
        assertEquals(Verdict.REJECT, result.verdict)
        assertEquals(Severity.LOW, result.severity)
    }

    @Test
    fun highSeverityForcesRejectBetweenCuts() {
        val result = DecisionCore.decide(0.5f, 0.06f, thresholds)
        assertEquals(Verdict.REJECT, result.verdict)
        assertEquals(Severity.HIGH, result.severity)
    }

    @Test
    fun safetyRelevantTypeBumpsSeverityOneLevel() {
        val result = DecisionCore.decide(0.5f, 0.02f, thresholds, DefectType.CRACK_OR_SCRATCH)
        assertEquals(Severity.HIGH, result.severity)
        assertEquals(Verdict.REJECT, result.verdict)
        val deformation = DecisionCore.decide(0.5f, 0.02f, thresholds, DefectType.DEFORMATION)
        assertEquals(Severity.MEDIUM, deformation.severity)
    }

    @Test
    fun confidenceIsHalfOnThresholdAndGrowsWithDistance() {
        val on = DecisionCore.decide(0.2f, 0f, thresholds).confidence
        val far = DecisionCore.decide(0.0f, 0f, thresholds).confidence
        assertEquals(0.5f, on, 1e-4f)
        assertTrue(far > on)
    }

    @Test(expected = IllegalArgumentException::class)
    fun rejectsNonFiniteScore() {
        DecisionCore.decide(Float.NaN, 0f, thresholds)
    }
}
