import { useState, useEffect } from "react";
import { Sparkles, Upload, Music, Loader2, Clock } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { toast } from "sonner";
import api from "@/lib/api";

export default function InputEditor({ projectId, project, sourceInput, setSourceInput, onInterpret, interpreting }) {
  const [rawText, setRawText] = useState(sourceInput?.raw_text || "");
  const [saving, setSaving] = useState(false);
  const [languageHint, setLanguageHint] = useState("auto");
  const [cultureHint, setCultureHint] = useState("auto");
  const [uploadingAudio, setUploadingAudio] = useState(false);
  const [audioInfo, setAudioInfo] = useState(null);

  // Load any existing transcription so it persists across refresh
  useEffect(() => {
    if (!projectId) return;
    api.getAudio(projectId).then(setAudioInfo).catch(() => {});
  }, [projectId]);

  const handleAudioUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploadingAudio(true);
    try {
      const result = await api.uploadAudio(projectId, file);
      setAudioInfo(result);
      if (result.text && !rawText.trim()) {
        setRawText(result.text);
      }
      toast.success(`Transcribed: ${result.segment_count} segments, ${result.total_duration?.toFixed(1)}s`);
    } catch (e) {
      toast.error(e?.response?.data?.detail || "Audio upload failed");
    } finally {
      setUploadingAudio(false);
    }
  };

  const handleSaveInput = async () => {
    if (!rawText.trim()) {
      toast.error("Please paste or type your content");
      return;
    }
    setSaving(true);
    try {
      const result = await api.addInput(projectId, {
        raw_text: rawText,
        language_hint: languageHint,
        culture_hint: cultureHint,
      });
      setSourceInput(result);
      toast.success(`Input saved: ${result.line_count} lines, detected as ${result.detected_type}`);
    } catch (e) {
      toast.error("Failed to save input");
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="h-full flex flex-col overflow-hidden">
      {/* Panel Header */}
      <div className="px-6 py-4 border-b border-[#27272A] shrink-0">
        <p className="panel-overline mb-1">Source Content</p>
        <h2 className="font-heading text-2xl text-[#F3F4F6]">Add Your Source Material</h2>
        <p className="text-sm text-[#71717A] mt-1">
          Paste lyrics, a script, poem, story, or voiceover. Qaivid will analyze it and build your entire video pipeline.
        </p>
      </div>

      <div className="flex-1 overflow-y-auto p-6 space-y-5">
        {/* Text Input */}
        <Textarea
          data-testid="source-text-input"
          value={rawText}
          onChange={(e) => setRawText(e.target.value)}
          placeholder={"Paste your source content here...\n\nExamples:\n\n[Verse 1]\nCharkha mera rang da ni\nBirha da dard sunaave\n\n[Chorus]\nVe mahi, pardesiya...\n\n— or a script, poem, story, voiceover —"}
          className="min-h-[300px] bg-[#1E1E24] border-[#27272A] text-[#F3F4F6] placeholder:text-[#4a4a55] font-heading text-lg leading-relaxed resize-none focus:ring-[#D4AF37] focus:border-[#D4AF37]"
          rows={15}
        />

        {/* Hint settings */}
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="panel-overline mb-2 block">Language Hint</label>
            <Select value={languageHint} onValueChange={setLanguageHint}>
              <SelectTrigger data-testid="language-hint-select" className="bg-[#1E1E24] border-[#27272A] text-[#F3F4F6]">
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="bg-[#1E1E24] border-[#27272A]">
                {["auto", "punjabi", "urdu", "hindi", "english"].map(l => (
                  <SelectItem key={l} value={l} className="text-[#F3F4F6] focus:bg-[#27272E] focus:text-white">
                    {l.charAt(0).toUpperCase() + l.slice(1)}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div>
            <label className="panel-overline mb-2 block">Culture Hint</label>
            <Select value={cultureHint} onValueChange={setCultureHint}>
              <SelectTrigger data-testid="culture-hint-select" className="bg-[#1E1E24] border-[#27272A] text-[#F3F4F6]">
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="bg-[#1E1E24] border-[#27272A]">
                {["auto", "punjabi_rural_lament", "punjabi_diaspora_memory", "urdu_philosophical_ghazal",
                  "devotional_qawwali", "north_indian_folk_female", "modern_urban_alienation", "generic_english"
                ].map(c => (
                  <SelectItem key={c} value={c} className="text-[#F3F4F6] focus:bg-[#27272E] focus:text-white">
                    {c.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>

        {/* Audio Upload */}
        <div className="bg-[#141417] border border-dashed border-[#27272A] rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Music className="w-5 h-5 text-[#D4AF37]" />
              <div>
                <p className="text-sm font-medium text-[#F3F4F6]">Upload Audio Track</p>
                <p className="text-[10px] text-[#4a4a55]">MP3, WAV, M4A — auto-transcribes with Whisper and syncs to timeline</p>
              </div>
            </div>
            <label className="cursor-pointer" data-testid="audio-upload-btn">
              <input type="file" accept=".mp3,.wav,.m4a,.mp4,.mpeg,.mpga,.webm" className="hidden" onChange={handleAudioUpload} disabled={uploadingAudio} />
              <span className={`inline-flex items-center gap-1.5 px-4 py-2 rounded-lg text-xs font-semibold transition-colors ${
                uploadingAudio ? 'bg-[#1E1E24] text-[#4a4a55]' : 'bg-[#1E1E24] text-[#A1A1AA] hover:bg-[#27272E] hover:text-white border border-[#27272A]'
              }`}>
                {uploadingAudio ? <><Loader2 className="w-3.5 h-3.5 animate-spin" />Transcribing...</> : <><Music className="w-3.5 h-3.5" />Upload Audio</>}
              </span>
            </label>
          </div>
          {audioInfo && (
            <div className="mt-3 space-y-2 animate-fade-up">
              <div className="bg-[#1E1E24] rounded p-3 text-xs text-[#A1A1AA] flex flex-wrap items-center gap-x-2 gap-y-1">
                {audioInfo.audio_filename && (
                  <>
                    <span className="text-[#D4AF37] font-medium">{audioInfo.audio_filename}</span>
                    <span className="text-[#4a4a55]">&middot;</span>
                  </>
                )}
                <span>{audioInfo.segment_count || (audioInfo.lines?.length ?? 0)} lines</span>
                <span className="text-[#4a4a55]">&middot;</span>
                <span>{audioInfo.total_duration?.toFixed(1) || 0}s</span>
                {audioInfo.language && (
                  <>
                    <span className="text-[#4a4a55]">&middot;</span>
                    <span>{audioInfo.language}</span>
                  </>
                )}
                {audioInfo.vocal_gender && audioInfo.vocal_gender !== "unknown" && (
                  <>
                    <span className="text-[#4a4a55]">&middot;</span>
                    <span>vocal: {audioInfo.vocal_gender}{audioInfo.vocal_age_range && audioInfo.vocal_age_range !== "unknown" ? `, ${audioInfo.vocal_age_range}` : ""}</span>
                  </>
                )}
              </div>

              {/* Transcribed lyrics with timestamps */}
              {(audioInfo.segments?.length > 0 || audioInfo.lines?.length > 0) && (
                <div className="bg-[#141417] border border-[#27272A] rounded-lg overflow-hidden" data-testid="transcription-lines">
                  <div className="px-3 py-2 border-b border-[#27272A] flex items-center justify-between">
                    <div className="flex items-center gap-1.5">
                      <Clock className="w-3 h-3 text-[#D4AF37]" />
                      <p className="panel-overline">Transcribed Lyrics</p>
                    </div>
                    <p className="text-[10px] text-[#4a4a55]">click anywhere to edit above</p>
                  </div>
                  <div className="max-h-[260px] overflow-y-auto py-1">
                    {(audioInfo.segments && audioInfo.segments.length > 0
                      ? audioInfo.segments.map((s, i) => ({ ts: s.start || 0, end: s.end, text: s.text, hint: s.segmentHint }))
                      : audioInfo.lines.map((l, i) => ({ ts: parseLineTimestamp(l.timestamp), text: l.line, hint: l.segmentHint }))
                    ).map((row, i) => (
                      <div key={i} className="flex items-start gap-3 px-3 py-1.5 hover:bg-[#1E1E24] transition-colors text-xs">
                        <span className="text-[10px] font-mono text-[#D4AF37] shrink-0 mt-0.5 w-12 text-right tabular-nums">
                          {formatTime(row.ts)}
                        </span>
                        {row.hint && row.hint !== "verse" && (
                          <span className="text-[9px] uppercase tracking-wide text-[#71717A] bg-[#1E1E24] px-1.5 py-0.5 rounded shrink-0 mt-0.5">{row.hint}</span>
                        )}
                        <p className="text-[#F3F4F6] font-heading flex-1">{row.text}</p>
                        {row.end && row.end > row.ts && (
                          <span className="text-[9px] text-[#3f3f46] font-mono shrink-0 mt-1 tabular-nums">{(row.end - row.ts).toFixed(1)}s</span>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Detection results */}
        {sourceInput && (
          <div className="bg-[#141417] border border-[#27272A] rounded-lg p-4 animate-fade-up" data-testid="detection-results">
            <p className="panel-overline mb-3">Detection Results</p>
            <div className="grid grid-cols-3 gap-4 text-sm">
              <div>
                <span className="text-[#71717A]">Type:</span>
                <span className="ml-2 text-[#D4AF37] font-medium">{sourceInput.detected_type}</span>
              </div>
              <div>
                <span className="text-[#71717A]">Language:</span>
                <span className="ml-2 text-[#F3F4F6]">{sourceInput.detected_language}</span>
              </div>
              <div>
                <span className="text-[#71717A]">Lines:</span>
                <span className="ml-2 text-[#F3F4F6]">{sourceInput.line_count}</span>
              </div>
            </div>
            {sourceInput.sections?.length > 0 && (
              <div className="mt-3">
                <span className="text-[#71717A] text-sm">Sections: </span>
                {sourceInput.sections.map((s, i) => (
                  <span key={i} className="inline-block text-xs bg-[#1E1E24] text-[#A1A1AA] px-2 py-0.5 rounded mr-1 mt-1">
                    {s.type} ({s.lines?.length || 0} lines)
                  </span>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Action buttons */}
        <div className="flex gap-3">
          <Button
            data-testid="save-input-btn"
            onClick={handleSaveInput}
            disabled={saving || !rawText.trim()}
            className="bg-[#1E1E24] border border-[#27272A] text-[#F3F4F6] hover:bg-[#27272E] hover:text-white"
          >
            <Upload className="w-4 h-4 mr-2" />
            {saving ? "Saving..." : "Save Input"}
          </Button>
          <Button
            data-testid="interpret-btn"
            onClick={onInterpret}
            disabled={interpreting || !sourceInput}
            className="bg-[#D4AF37] text-[#0A0A0C] hover:bg-[#F1C40F] hover:text-[#0A0A0C] font-semibold"
          >
            <Sparkles className="w-4 h-4 mr-2" />
            {interpreting ? "Analyzing..." : "Run Context Intelligence"}
          </Button>
        </div>

        {interpreting && (
          <div className="bg-[#141417] border border-[#D4AF37]/20 rounded-lg p-4 gold-glow">
            <p className="text-[#D4AF37] text-sm font-medium loading-pulse">
              Context engine is analyzing meaning, culture, emotion, and narrative structure...
            </p>
            <p className="text-[#4a4a55] text-xs mt-1">This powers the entire video pipeline downstream — scenes, shots, and generation prompts.</p>
          </div>
        )}
      </div>
    </div>
  );
}

function formatTime(sec) {
  if (sec === undefined || sec === null || isNaN(sec)) return "0:00";
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

function parseLineTimestamp(ts) {
  if (typeof ts === "number") return ts;
  if (!ts) return 0;
  const clean = String(ts).replace(/[\[\]]/g, "").trim();
  const m = clean.match(/^(\d+):(\d{2})(?:[.,](\d+))?$/);
  if (!m) return 0;
  return parseInt(m[1], 10) * 60 + parseInt(m[2], 10) + parseInt((m[3] || "0").padEnd(3, "0").slice(0, 3), 10) / 1000;
}
