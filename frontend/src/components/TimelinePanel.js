import { useState, useEffect, useRef } from "react";
import { Clock, Play, Pause, Music, Film, ChevronRight } from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import api from "@/lib/api";

const BACKEND = process.env.REACT_APP_BACKEND_URL;

export default function TimelinePanel({ projectId }) {
  const [audio, setAudio] = useState(null);
  const [scenes, setScenes] = useState([]);
  const [shots, setShots] = useState([]);
  const [sourceInput, setSourceInput] = useState(null);
  const [timeline, setTimeline] = useState(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [playing, setPlaying] = useState(false);
  const audioRef = useRef(null);
  const intervalRef = useRef(null);

  useEffect(() => {
    api.getAudio(projectId).then(setAudio).catch(() => {});
    api.getScenes(projectId).then(setScenes).catch(() => {});
    api.getShots(projectId).then(setShots).catch(() => {});
    api.getInput(projectId).then(setSourceInput).catch(() => {});
    api.getTimeline(projectId).then(setTimeline).catch(() => {});
  }, [projectId]);

  useEffect(() => {
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, []);

  const togglePlay = () => {
    if (!audioRef.current) return;
    if (playing) {
      audioRef.current.pause();
      if (intervalRef.current) clearInterval(intervalRef.current);
    } else {
      audioRef.current.play();
      intervalRef.current = setInterval(() => {
        if (audioRef.current) setCurrentTime(audioRef.current.currentTime);
      }, 100);
    }
    setPlaying(!playing);
  };

  const seekTo = (time) => {
    if (audioRef.current) { audioRef.current.currentTime = time; setCurrentTime(time); }
  };

  const segments = audio?.segments || [];
  const timedSegments = sourceInput?.timed_segments || [];
  const hasTimestamps = segments.length > 0 || timedSegments.length > 0;
  const displaySegments = segments.length > 0 ? segments : timedSegments;
  const totalDuration = audio?.total_duration || (displaySegments.length > 0 ? Math.max(...displaySegments.filter(s => s.end).map(s => s.end), 0) : 0);

  // Build shot timeline bars
  const shotBars = [];
  let shotOffset = 0;
  for (const shot of shots) {
    const dur = shot.duration_hint || 3;
    shotBars.push({ ...shot, start: shotOffset, end: shotOffset + dur });
    shotOffset += dur;
  }
  const shotTotalDur = shotOffset || 1;

  return (
    <div className="h-full flex flex-col">
      <div className="px-6 py-4 border-b border-[#27272A] shrink-0">
        <p className="panel-overline mb-1">Timeline</p>
        <h2 className="font-heading text-2xl text-[#F3F4F6]">Audio & Shot Timeline</h2>
        <p className="text-sm text-[#71717A] mt-1">
          {hasTimestamps ? "Synced audio timeline with transcription segments and shot plan." : "Upload audio on the Source panel to enable timestamped timeline."}
        </p>
      </div>

      <ScrollArea className="flex-1">
        <div className="p-6 space-y-6">

          {/* Audio player */}
          {audio?.audio_filepath && (
            <div className="bg-[#141417] border border-[#27272A] rounded-xl p-4" data-testid="audio-player">
              <div className="flex items-center gap-3 mb-3">
                <Button onClick={togglePlay} size="icon" className="w-10 h-10 rounded-full bg-[#D4AF37] text-[#0A0A0C] hover:bg-[#F1C40F]" data-testid="play-btn">
                  {playing ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4 ml-0.5" />}
                </Button>
                <div className="flex-1">
                  <div className="flex items-center justify-between text-xs mb-1">
                    <span className="text-[#F3F4F6] font-mono">{formatTime(currentTime)}</span>
                    <span className="text-[#4a4a55] font-mono">{formatTime(totalDuration)}</span>
                  </div>
                  {/* Waveform-style progress */}
                  <div className="h-8 bg-[#1E1E24] rounded-lg overflow-hidden relative cursor-pointer"
                    onClick={e => { const r = e.currentTarget.getBoundingClientRect(); seekTo((e.clientX - r.left) / r.width * totalDuration); }}>
                    {/* Segment blocks */}
                    {displaySegments.map((seg, i) => {
                      if (!seg.start && seg.start !== 0) return null;
                      const left = ((seg.start || 0) / Math.max(totalDuration, 1)) * 100;
                      const width = (((seg.end || seg.start + 2) - (seg.start || 0)) / Math.max(totalDuration, 1)) * 100;
                      const isActive = currentTime >= (seg.start || 0) && currentTime < (seg.end || seg.start + 2);
                      return (
                        <div key={i} className={`absolute top-0 h-full rounded-sm transition-colors ${isActive ? 'bg-[#D4AF37]/40' : 'bg-[#D4AF37]/15'}`}
                          style={{ left: `${left}%`, width: `${Math.max(width, 0.5)}%` }}
                          onClick={e => { e.stopPropagation(); seekTo(seg.start || 0); }} />
                      );
                    })}
                    {/* Playhead */}
                    <div className="absolute top-0 w-0.5 h-full bg-[#D4AF37] transition-all" style={{ left: `${(currentTime / Math.max(totalDuration, 1)) * 100}%` }} />
                  </div>
                </div>
                <Music className="w-5 h-5 text-[#D4AF37] shrink-0" />
              </div>
              <audio ref={audioRef} src={`${BACKEND}/api/reference-images/${audio.audio_filepath?.split('/').pop()}`} onEnded={() => setPlaying(false)} className="hidden" />
              <p className="text-[10px] text-[#4a4a55]">{audio.audio_filename} &middot; {audio.language} &middot; {audio.segment_count} segments</p>
            </div>
          )}

          {/* Transcription segments */}
          {displaySegments.length > 0 && (
            <div data-testid="transcript-segments">
              <p className="panel-overline mb-3">Transcription Timeline</p>
              <div className="space-y-1">
                {displaySegments.map((seg, i) => {
                  const isActive = currentTime >= (seg.start || 0) && currentTime < (seg.end || (seg.start || 0) + 2);
                  return (
                    <div key={i} onClick={() => seekTo(seg.start || 0)}
                      className={`flex items-start gap-3 px-3 py-2 rounded-lg cursor-pointer transition-all ${
                        isActive ? 'bg-[#D4AF37]/10 border border-[#D4AF37]/20' : 'hover:bg-[#1E1E24] border border-transparent'
                      }`} data-testid={`segment-${i}`}>
                      <span className={`text-[10px] font-mono shrink-0 mt-0.5 w-12 ${isActive ? 'text-[#D4AF37]' : 'text-[#4a4a55]'}`}>
                        {formatTime(seg.start || 0)}
                      </span>
                      <p className={`text-sm font-heading ${isActive ? 'text-[#F3F4F6]' : 'text-[#A1A1AA]'}`}>{seg.text}</p>
                      {seg.end && <span className="text-[9px] text-[#3f3f46] font-mono shrink-0 mt-0.5">{((seg.end - (seg.start || 0))).toFixed(1)}s</span>}
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Shot timeline */}
          {shotBars.length > 0 && (
            <div data-testid="shot-timeline">
              <p className="panel-overline mb-3">Shot Timeline</p>
              {/* Visual bar */}
              <div className="h-12 bg-[#1E1E24] rounded-lg overflow-hidden relative mb-3">
                {shotBars.map((sb, i) => {
                  const left = (sb.start / shotTotalDur) * 100;
                  const width = ((sb.end - sb.start) / shotTotalDur) * 100;
                  const colors = ["bg-[#D4AF37]", "bg-[#C85A17]", "bg-[#8B5CF6]", "bg-[#3B82F6]", "bg-[#10B981]", "bg-[#EF4444]"];
                  return (
                    <div key={i} className={`absolute top-0 h-full ${colors[i % colors.length]}/30 border-r border-[#0A0A0C] flex items-center justify-center`}
                      style={{ left: `${left}%`, width: `${Math.max(width, 1)}%` }}
                      title={`Shot ${sb.shot_number}: ${sb.duration_hint}s`}>
                      <span className="text-[8px] text-white/60 font-mono">{sb.shot_number}</span>
                    </div>
                  );
                })}
              </div>
              {/* Shot list */}
              <div className="space-y-1">
                {shotBars.map((sb, i) => (
                  <div key={i} className="flex items-center gap-3 px-3 py-1.5 text-xs rounded hover:bg-[#1E1E24] transition-colors">
                    <span className="text-[10px] font-mono text-[#4a4a55] w-10">{formatTime(sb.start)}</span>
                    <span className="text-[9px] bg-[#1E1E24] text-[#71717A] px-1.5 py-0.5 rounded">Shot {sb.shot_number}</span>
                    <span className="text-[#D4AF37] text-[10px]">{sb.shot_type}</span>
                    <span className="text-[#A1A1AA] flex-1 truncate">{sb.visual_priority}</span>
                    <span className="text-[10px] text-[#4a4a55] font-mono">{sb.duration_hint}s</span>
                  </div>
                ))}
              </div>
              <div className="mt-2 text-right">
                <span className="text-[10px] text-[#4a4a55]">Total: {shotTotalDur.toFixed(1)}s &middot; {shotBars.length} shots</span>
              </div>
            </div>
          )}

          {/* Scene blocks */}
          {scenes.length > 0 && (
            <div data-testid="scene-blocks">
              <p className="panel-overline mb-3">Scene Blocks</p>
              <div className="flex gap-2 flex-wrap">
                {scenes.map((sc, i) => (
                  <div key={i} className="bg-[#141417] border border-[#27272A] rounded-lg px-4 py-3 min-w-[150px]">
                    <div className="flex items-center gap-1.5 mb-1">
                      <Film className="w-3 h-3 text-[#D4AF37]" />
                      <span className="text-[10px] text-[#D4AF37] font-semibold">Scene {sc.scene_number}</span>
                    </div>
                    <p className="text-[10px] text-[#A1A1AA] line-clamp-1">{sc.purpose?.split('—')[0]}</p>
                    <p className="text-[9px] text-[#4a4a55] mt-0.5">{sc.temporal_status} &middot; {sc.emotional_temperature || 'neutral'}</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Empty state */}
          {!hasTimestamps && shotBars.length === 0 && scenes.length === 0 && (
            <div className="text-center py-16">
              <Clock className="w-12 h-12 text-[#27272A] mx-auto mb-4" />
              <h3 className="text-lg text-[#A1A1AA] mb-2">No timeline data</h3>
              <p className="text-sm text-[#4a4a55]">Upload audio on the Source panel for timestamped segments, or build scenes and shots to see the shot timeline.</p>
            </div>
          )}
        </div>
      </ScrollArea>
    </div>
  );
}

function formatTime(sec) {
  if (!sec && sec !== 0) return "0:00";
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}
