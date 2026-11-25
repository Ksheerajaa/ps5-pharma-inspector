package com.example.ps5.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Shapes
import androidx.compose.material3.Text
import androidx.compose.material3.Typography
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import java.time.LocalTime
import java.time.format.DateTimeFormatter

/** Monochrome industrial palette with a single signal-orange accent. */
val Ink = Color(0xFF111111)
val Paper = Color(0xFFEDEDED)
val CardWhite = Color(0xFFFFFFFF)
val Muted = Color(0xFF8A8A8A)
val Signal = Color(0xFFFF4B1F)

private val Colours = lightColorScheme(
    primary = Ink, onPrimary = Color.White, primaryContainer = Color(0xFFDADADA), onPrimaryContainer = Ink,
    secondary = Signal, onSecondary = Color.White,
    background = Paper, surface = Paper, surfaceVariant = Color(0xFFE2E2E2),
    onSurface = Ink, onSurfaceVariant = Color(0xFF5E5E5E), outline = Color(0xFFCFCFCF),
)

private val Type = Typography(
    headlineLarge = TextStyle(fontSize = 34.sp, fontWeight = FontWeight.Light, letterSpacing = (-1).sp),
    headlineMedium = TextStyle(fontSize = 26.sp, fontWeight = FontWeight.Light, letterSpacing = (-0.5).sp),
    titleLarge = TextStyle(fontSize = 20.sp, fontWeight = FontWeight.Normal),
    labelSmall = TextStyle(fontSize = 10.sp, fontWeight = FontWeight.Medium, letterSpacing = 1.sp),
)

@Composable
fun Ps5Theme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = Colours, typography = Type,
        shapes = Shapes(small = RoundedCornerShape(12.dp), medium = RoundedCornerShape(20.dp), large = RoundedCornerShape(28.dp)),
        content = content,
    )
}

@Composable
private fun Pill(text: String, dot: Color) {
    Row(
        Modifier.clip(RoundedCornerShape(50)).background(Color(0xFF222222)).padding(horizontal = 10.dp, vertical = 5.dp),
        verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp),
    ) {
        Box(Modifier.size(7.dp).clip(CircleShape).background(dot))
        Text(text, color = Color.White, fontSize = 11.sp, fontWeight = FontWeight.Medium)
    }
}

/** Black header card: live time, title, and the facts an operator or judge wants first. */
@Composable
fun AppHeader(product: String, llamaReady: Boolean, bank: Int, onLongPress: () -> Unit = {}) {
    Column(
        Modifier.fillMaxWidth().clip(RoundedCornerShape(bottomStart = 28.dp, bottomEnd = 28.dp)).background(Ink)
            .padding(start = 20.dp, end = 20.dp, top = 40.dp, bottom = 16.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            Box(Modifier.size(7.dp).clip(CircleShape).background(Signal))
            Text("${LocalTime.now().format(DateTimeFormatter.ofPattern("HH:mm"))} · Station 1 · $product",
                color = Color(0xFFBDBDBD), style = MaterialTheme.typography.labelSmall)
        }
        Text("Pharma Inspector", color = Color.White, style = MaterialTheme.typography.headlineMedium,
            modifier = Modifier.pointerInput(Unit) { detectTapGestures(onLongPress = { onLongPress() }) })
        Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            Pill("Offline", Color(0xFF4CD07D))
            Pill(if (llamaReady) "Llama on-device" else "Template advice", if (llamaReady) Color(0xFF4CD07D) else Signal)
            Pill("Bank $bank", Color.White)
        }
    }
}
