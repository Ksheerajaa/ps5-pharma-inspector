package com.example.ps5.core

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class AnomalyAnalysisTest {
    private val size = 32

    private fun map(fill: (Int, Int) -> Boolean) = FloatArray(size * size) { if (fill(it % size, it / size)) 5f else 0f }

    @Test
    fun emptyMapHasNoBlob() {
        val f = AnomalyAnalysis.analyse(FloatArray(size * size), size, 1f)
        assertEquals(0f, f.areaFraction)
        assertNull(f.box)
        assertEquals("none", f.location)
    }

    @Test
    fun largestBlobBoxAndLocation() {
        val m = map { x, y -> (x in 2..5 && y in 2..5) || (x == 30 && y == 30) }
        val f = AnomalyAnalysis.analyse(m, size, 1f)
        assertEquals(2, f.blobCount)
        assertArrayEquals(intArrayOf(2, 2, 4, 4), f.box)
        assertEquals("upper-left", f.location)
        assertEquals(17f / (size * size), f.areaFraction, 1e-6f)
    }

    @Test
    fun thinLineIsElongatedAndTypedAsCrack() {
        val f = AnomalyAnalysis.analyse(map { x, y -> y == 16 && x in 4..27 }, size, 1f)
        assertTrue(f.elongation > DefectTyper.ELONGATION_CRACK)
        assertEquals("centre", f.location)
        assertEquals(DefectType.CRACK_OR_SCRATCH, DefectTyper.classify(f, 0.0, 0.5f).type)
    }

    @Test
    fun compactColourBlobIsContamination() {
        val f = AnomalyAnalysis.analyse(map { x, y -> x in 10..14 && y in 10..14 }, size, 1f)
        assertEquals(DefectType.CONTAMINATION_OR_COLOUR, DefectTyper.classify(f, 20.0, 0.5f).type)
        assertEquals(DefectType.DEFORMATION, DefectTyper.classify(f, 1.0, 0.01f).type)
    }
}
