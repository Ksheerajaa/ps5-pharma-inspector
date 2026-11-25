package com.example.ps5

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.PrimaryScrollableTabRow
import androidx.compose.material3.Surface
import androidx.compose.material3.Tab
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.example.ps5.ui.AboutScreen
import com.example.ps5.ui.AppHeader
import com.example.ps5.ui.DemoScript
import com.example.ps5.ui.Ps5Theme
import com.example.ps5.ui.BenchScreen
import com.example.ps5.ui.HistoryScreen
import com.example.ps5.ui.InspectScreen
import com.example.ps5.ui.ReplayScreen

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            Ps5Theme {
                Surface(modifier = Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) { App() }
            }
        }
    }

    override fun onTrimMemory(level: Int) {
        super.onTrimMemory(level)
        if (level >= TRIM_MEMORY_RUNNING_CRITICAL) AppEngine.get(this).llama.unload()
    }
}

@Composable
private fun App() {
    val context = LocalContext.current
    val engine = remember { AppEngine.get(context) }
    var tab by remember { mutableIntStateOf(0) }
    var demoFile by remember { mutableStateOf<String?>(null) }
    // Demo walkthrough: launch with `--ez demo true`, or long-press the title on the phone.
    var demoRun by remember { mutableIntStateOf(if ((context as? android.app.Activity)?.intent?.getBooleanExtra("demo", false) == true) 1 else 0) }
    LaunchedEffect(demoRun) { if (demoRun > 0) DemoScript.run({ tab = it }, { demoFile = it }) }
    val tabs = listOf("Inspect", "Replay", "History", "Bench", "About")
    Column(Modifier.fillMaxSize()) {
        AppHeader(engine.productLabel(), engine.llama.filesPresent(), engine.bankSize, onLongPress = { demoFile = null; demoRun++ })
        PrimaryScrollableTabRow(
            selectedTabIndex = tab, edgePadding = 4.dp,
            containerColor = MaterialTheme.colorScheme.surface, contentColor = MaterialTheme.colorScheme.primary,
        ) {
            tabs.forEachIndexed { i, t ->
                Tab(selected = tab == i, onClick = { tab = i }, text = { Text(t, fontWeight = if (tab == i) FontWeight.Bold else FontWeight.Normal) })
            }
        }
        val m = Modifier.fillMaxSize().padding(horizontal = 12.dp, vertical = 10.dp)
        when (tab) {
            0 -> InspectScreen(engine, m)
            1 -> ReplayScreen(engine, m, demoFile)
            2 -> HistoryScreen(engine, m)
            3 -> BenchScreen(engine, m)
            else -> AboutScreen(engine, m)
        }
    }
}
