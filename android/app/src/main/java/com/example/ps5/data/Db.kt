package com.example.ps5.data

import android.content.Context
import androidx.room.Dao
import androidx.room.Database
import androidx.room.Entity
import androidx.room.Insert
import androidx.room.PrimaryKey
import androidx.room.Query
import androidx.room.Room
import androidx.room.RoomDatabase
import androidx.room.Transaction
import kotlinx.coroutines.flow.Flow

/** One row per item (or one per frame for NO_ITEMS / BAD_IMAGE). Plan 6.8 + addendum 2.4. */
@Entity(tableName = "inspections")
data class InspectionEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val tsUtc: String,
    val sessionId: String,
    val source: String,            // camera or replay:<file>
    val product: String,
    val frameId: String,
    val itemIdx: Int?,
    val nItemsInFrame: Int,
    val status: String,            // OK, NO_ITEMS, BAD_IMAGE, CLUSTER, PARTIAL, UNKNOWN_OBJECT, TOO_MANY
    val statusReason: String?,
    val referenceId: String?,
    val score: Float?,
    val thresholdPass: Float?,
    val thresholdReject: Float?,
    val sensitivity: Int = 0,
    val verdict: String?,
    val severity: String?,
    val defectType: String?,
    val typeBasis: String?,
    val areaPct: Float?,
    val bbox: String?,             // item box in frame pixels x,y,w,h
    val defectBox: String?,        // anomaly box in crop pixels x,y,w,h
    val location: String?,
    val confidence: Float?,
    val handoffJson: String?,
    val advisoryAction: String? = null,
    val advisoryReason: String? = null,
    val advisoryCheck: String? = null,
    val logText: String? = null,
    val advisorySource: String? = null, // llm or template
    val advisoryNote: String? = null,   // queue_full, timeout, validator:<reason>, model_missing
    val queueWaitMs: Long? = null,
    val visionMs: Long?,
    val preprocessMs: Long?,
    val frameMs: Long?,
    val llmLoadMs: Long? = null,
    val llmTtftMs: Long? = null,
    val llmTokens: Int? = null,
    val llmTotalMs: Long? = null,
    val llmRaw: String? = null,
    val thumbPath: String?,
    val modelVersions: String,
)

@Dao
interface InspectionDao {
    @Insert
    suspend fun insertAll(rows: List<InspectionEntity>): List<Long>

    @Query("SELECT * FROM inspections ORDER BY id DESC LIMIT :limit")
    fun recent(limit: Int): Flow<List<InspectionEntity>>

    @Query("SELECT * FROM inspections ORDER BY id ASC")
    suspend fun all(): List<InspectionEntity>

    @Query("SELECT * FROM inspections WHERE id = :id")
    suspend fun byId(id: Long): InspectionEntity?

    @Query("SELECT COUNT(*) FROM (SELECT verdict FROM inspections WHERE verdict IS NOT NULL ORDER BY id DESC LIMIT 20) WHERE verdict = 'REJECT'")
    suspend fun rejectsInLast20(): Int

    @Query(
        "UPDATE inspections SET advisoryAction = :action, advisoryReason = :reason, advisoryCheck = :check, " +
            "logText = :log, advisorySource = :source, advisoryNote = :note, queueWaitMs = :waitMs, " +
            "llmLoadMs = :loadMs, llmTtftMs = :ttftMs, llmTokens = :tokens, llmTotalMs = :totalMs, llmRaw = :raw WHERE id = :id",
    )
    suspend fun updateAdvisory(
        id: Long, action: String, reason: String, check: String, log: String, source: String, note: String?,
        waitMs: Long?, loadMs: Long?, ttftMs: Long?, tokens: Int?, totalMs: Long?, raw: String?,
    )

    @Query("DELETE FROM inspections")
    suspend fun clear()
}

@Database(entities = [InspectionEntity::class], version = 1, exportSchema = false)
abstract class AppDb : RoomDatabase() {
    abstract fun dao(): InspectionDao

    companion object {
        @Volatile private var instance: AppDb? = null
        fun get(context: Context): AppDb = instance ?: synchronized(this) {
            instance ?: Room.databaseBuilder(context.applicationContext, AppDb::class.java, "inspections.db")
                .build().also { instance = it }
        }
    }
}

object Csv {
    fun export(rows: List<InspectionEntity>): String {
        val fields = InspectionEntity::class.java.declaredFields.filter { !java.lang.reflect.Modifier.isStatic(it.modifiers) }
        fields.forEach { it.isAccessible = true }
        return buildString {
            appendLine(fields.joinToString(",") { it.name })
            for (row in rows) {
                appendLine(fields.joinToString(",") { f -> escape(f.get(row)?.toString() ?: "") })
            }
        }
    }

    fun escape(v: String): String =
        if (v.any { it == ',' || it == '"' || it == '\n' }) "\"" + v.replace("\"", "\"\"") + "\"" else v
}
