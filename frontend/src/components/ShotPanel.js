import { useState, useEffect } from "react";
import { DndContext, closestCenter, PointerSensor, useSensor, useSensors } from "@dnd-kit/core";
import { SortableContext, verticalListSortingStrategy, useSortable } from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";
import { Camera, Video, Sun, Move, FileText, GripVertical, Pencil, Check, X } from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { toast } from "sonner";
import api from "@/lib/api";

const SHOT_TYPES = ["wide","medium","medium-close","close-up","extreme-close-up","over-shoulder","aerial","pov"];
const CAMERA_HEIGHTS = ["eye-level","low-angle","high-angle","overhead","dutch-tilt"];
const CAMERA_BEHAVIORS = ["static","slow-pan-left","slow-pan-right","track-forward","dolly-in","dolly-out","handheld","crane-up","orbit"];

function SortableShot({ shot, idx, projectId, onShotUpdate }) {
  const { attributes, listeners, setNodeRef, transform, transition, isDragging } = useSortable({ id: shot.id });
  const style = { transform: CSS.Transform.toString(transform), transition, opacity: isDragging ? 0.5 : 1, zIndex: isDragging ? 50 : 'auto' };
  const [editing, setEditing] = useState(false);
  const [edits, setEdits] = useState({});

  const startEdit = () => {
    setEdits({
      shot_type: shot.shot_type || "medium",
      camera_height: shot.camera_height || "eye-level",
      camera_behavior: shot.camera_behavior || "static",
      subject_action: shot.subject_action || "",
      emotional_micro_state: shot.emotional_micro_state || "",
      light_description: shot.light_description || "",
      duration_hint: shot.duration_hint || 3,
    });
    setEditing(true);
  };

  const saveEdit = async () => {
    try {
      const updated = await api.updateShot(projectId, shot.id, edits);
      onShotUpdate(updated);
      setEditing(false);
      toast.success("Shot updated");
    } catch { toast.error("Update failed"); }
  };

  return (
    <div ref={setNodeRef} style={style} data-testid={`shot-card-${shot.shot_number}`}
      className="film-slate bg-[#141417] border border-[#27272A] rounded-lg p-4 animate-fade-up" {...attributes}>
      <div className="flex items-start gap-2.5 mb-3">
        <button {...listeners} className="mt-0.5 p-1 rounded hover:bg-[#27272E] text-[#3f3f46] hover:text-[#D4AF37] cursor-grab active:cursor-grabbing shrink-0" data-testid={`drag-shot-${idx}`}>
          <GripVertical className="w-3.5 h-3.5" />
        </button>
        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="text-[10px] tracking-widest uppercase text-[#71717A] font-semibold bg-[#1E1E24] px-2 py-0.5 rounded">Shot {shot.shot_number}</span>
              {editing ? (
                <Select value={edits.shot_type} onValueChange={v => setEdits(p => ({...p, shot_type: v}))}>
                  <SelectTrigger className="bg-[#1E1E24] border-[#27272A] text-[#D4AF37] text-[10px] h-6 w-28"><SelectValue /></SelectTrigger>
                  <SelectContent className="bg-[#1E1E24] border-[#27272A]">{SHOT_TYPES.map(t => <SelectItem key={t} value={t} className="text-[#F3F4F6] text-xs">{t}</SelectItem>)}</SelectContent>
                </Select>
              ) : (
                <span className="text-xs text-[#D4AF37] bg-[#D4AF37]/10 px-2 py-0.5 rounded font-medium">{shot.shot_type}</span>
              )}
            </div>
            <div className="flex items-center gap-1.5">
              {editing ? (
                <Input type="number" value={edits.duration_hint} onChange={e => setEdits(p => ({...p, duration_hint: parseFloat(e.target.value) || 3}))}
                  className="bg-[#1E1E24] border-[#27272A] text-[#F3F4F6] text-[10px] h-6 w-14 text-center font-mono" step="0.5" />
              ) : (
                <span className="text-[10px] text-[#71717A] font-mono">{shot.duration_hint}s</span>
              )}
              {editing ? (
                <>
                  <button onClick={saveEdit} className="p-1 rounded hover:bg-[#10B981]/20 text-[#10B981]" data-testid={`save-shot-${idx}`}><Check className="w-3.5 h-3.5" /></button>
                  <button onClick={() => setEditing(false)} className="p-1 rounded hover:bg-[#EF4444]/20 text-[#EF4444]"><X className="w-3.5 h-3.5" /></button>
                </>
              ) : (
                <button onClick={startEdit} className="p-1 rounded hover:bg-[#27272E] text-[#3f3f46] hover:text-[#D4AF37]" data-testid={`edit-shot-${idx}`}><Pencil className="w-3.5 h-3.5" /></button>
              )}
            </div>
          </div>
        </div>
      </div>
      <div className="ml-8">
        <p className="text-sm text-[#F3F4F6] mb-3 font-heading italic">{shot.visual_priority}</p>

        {editing ? (
          <div className="space-y-2 mb-3">
            <div className="grid grid-cols-2 gap-2">
              <Select value={edits.camera_height} onValueChange={v => setEdits(p => ({...p, camera_height: v}))}>
                <SelectTrigger className="bg-[#1E1E24] border-[#27272A] text-[#F3F4F6] text-xs h-8"><SelectValue placeholder="Camera height" /></SelectTrigger>
                <SelectContent className="bg-[#1E1E24] border-[#27272A]">{CAMERA_HEIGHTS.map(h => <SelectItem key={h} value={h} className="text-[#F3F4F6] text-xs">{h}</SelectItem>)}</SelectContent>
              </Select>
              <Select value={edits.camera_behavior} onValueChange={v => setEdits(p => ({...p, camera_behavior: v}))}>
                <SelectTrigger className="bg-[#1E1E24] border-[#27272A] text-[#F3F4F6] text-xs h-8"><SelectValue placeholder="Camera behavior" /></SelectTrigger>
                <SelectContent className="bg-[#1E1E24] border-[#27272A]">{CAMERA_BEHAVIORS.map(b => <SelectItem key={b} value={b} className="text-[#F3F4F6] text-xs">{b}</SelectItem>)}</SelectContent>
              </Select>
            </div>
            <Textarea value={edits.subject_action} onChange={e => setEdits(p => ({...p, subject_action: e.target.value}))}
              placeholder="Subject action" className="bg-[#1E1E24] border-[#27272A] text-[#F3F4F6] text-xs min-h-[50px]" rows={2} />
            <Input value={edits.emotional_micro_state} onChange={e => setEdits(p => ({...p, emotional_micro_state: e.target.value}))}
              placeholder="Emotional micro-state" className="bg-[#1E1E24] border-[#27272A] text-[#F3F4F6] text-xs h-8" />
            <Input value={edits.light_description} onChange={e => setEdits(p => ({...p, light_description: e.target.value}))}
              placeholder="Lighting" className="bg-[#1E1E24] border-[#27272A] text-[#F3F4F6] text-xs h-8" />
          </div>
        ) : (
          <>
            <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-xs">
              <div className="flex items-center gap-1.5"><Camera className="w-3 h-3 text-[#71717A]" /><span className="text-[#71717A]">Height:</span><span className="text-[#A1A1AA]">{shot.camera_height}</span></div>
              <div className="flex items-center gap-1.5"><Video className="w-3 h-3 text-[#71717A]" /><span className="text-[#71717A]">Camera:</span><span className="text-[#A1A1AA]">{shot.camera_behavior}</span></div>
              <div className="flex items-center gap-1.5"><Sun className="w-3 h-3 text-[#71717A]" /><span className="text-[#71717A]">Light:</span><span className="text-[#A1A1AA] truncate">{shot.light_description}</span></div>
              <div className="flex items-center gap-1.5"><Move className="w-3 h-3 text-[#71717A]" /><span className="text-[#71717A]">Motion:</span><span className="text-[#A1A1AA] truncate">{shot.motion_constraints}</span></div>
            </div>
            <div className="mt-3 bg-[#1E1E24] rounded p-2.5">
              <span className="text-[10px] text-[#71717A] uppercase tracking-wider">Action</span>
              <p className="text-xs text-[#F3F4F6] mt-1">{shot.subject_action}</p>
            </div>
            {shot.emotional_micro_state && <div className="mt-2 text-xs"><span className="text-[#71717A]">Emotion: </span><span className="text-[#C85A17]">{shot.emotional_micro_state}</span></div>}
            {shot.secondary_objects?.length > 0 && <div className="flex flex-wrap gap-1 mt-2">{shot.secondary_objects.map((obj, i) => <span key={i} className="text-[10px] bg-[#1E1E24] text-[#A1A1AA] px-2 py-0.5 rounded">{obj}</span>)}</div>}
          </>
        )}
      </div>
    </div>
  );
}

export default function ShotPanel({ shots, scenes, projectId, onBuildPrompts, buildingPrompts }) {
  const [modelTarget, setModelTarget] = useState("generic");
  const [models, setModels] = useState([]);
  const [localShots, setLocalShots] = useState(shots);
  const sensors = useSensors(useSensor(PointerSensor, { activationConstraint: { distance: 8 } }));

  useEffect(() => { api.listModels().then(setModels).catch(() => {}); }, []);
  useEffect(() => { setLocalShots(shots); }, [shots]);

  const handleDragEnd = async (event) => {
    const { active, over } = event;
    if (!over || active.id === over.id) return;
    const oldIdx = localShots.findIndex(s => s.id === active.id);
    const newIdx = localShots.findIndex(s => s.id === over.id);
    if (oldIdx === -1 || newIdx === -1) return;
    const reordered = [...localShots];
    const [moved] = reordered.splice(oldIdx, 1);
    reordered.splice(newIdx, 0, moved);
    const renumbered = reordered.map((s, i) => ({ ...s, shot_number: i + 1 }));
    setLocalShots(renumbered);
    try { await api.reorderShots(projectId, renumbered.map(s => s.id)); toast.success("Reordered"); }
    catch { toast.error("Failed"); setLocalShots(shots); }
  };

  const handleShotUpdate = (updated) => {
    setLocalShots(prev => prev.map(s => s.id === updated.id ? { ...s, ...updated } : s));
  };

  if (!localShots?.length) return <div className="h-full flex items-center justify-center text-[#71717A]">No shots yet. Build shots from the Storyboard panel.</div>;

  const sceneMap = {};
  scenes.forEach(s => { sceneMap[s.id] = s; });
  const grouped = {};
  localShots.forEach(shot => { const sid = shot.scene_id; if (!grouped[sid]) grouped[sid] = []; grouped[sid].push(shot); });

  return (
    <div className="h-full flex flex-col">
      <div className="px-6 py-4 border-b border-[#27272A] flex items-center justify-between shrink-0">
        <div>
          <p className="panel-overline mb-1">Cinematography</p>
          <h2 className="font-heading text-2xl text-[#F3F4F6]">{localShots.length} Shot{localShots.length !== 1 ? 's' : ''}</h2>
          <p className="text-[10px] text-[#4a4a55] mt-0.5">Drag to reorder. Click pencil to edit shot type, camera, action, lighting.</p>
        </div>
        <div className="flex items-center gap-3">
          <Select value={modelTarget} onValueChange={setModelTarget}>
            <SelectTrigger data-testid="model-target-select" className="w-36 bg-[#1E1E24] border-[#27272A] text-[#F3F4F6] text-xs h-8"><SelectValue /></SelectTrigger>
            <SelectContent className="bg-[#1E1E24] border-[#27272A]">
              {models.map(m => <SelectItem key={m.id} value={m.id} className="text-[#F3F4F6] focus:bg-[#27272E] text-xs">{m.id} {m.supports_motion ? "(video)" : "(image)"}</SelectItem>)}
              {models.length === 0 && <SelectItem value="generic" className="text-[#F3F4F6] text-xs">generic</SelectItem>}
            </SelectContent>
          </Select>
          <Button data-testid="build-prompts-btn" onClick={() => onBuildPrompts(modelTarget)} disabled={buildingPrompts}
            className="bg-[#D4AF37] text-[#0A0A0C] hover:bg-[#F1C40F] font-semibold text-xs">
            <FileText className="w-3.5 h-3.5 mr-1.5" />{buildingPrompts ? "Generating..." : "Generate Prompts"}
          </Button>
        </div>
      </div>
      <ScrollArea className="flex-1">
        <div className="p-6 space-y-6">
          {Object.entries(grouped).map(([sceneId, sceneShots]) => {
            const scene = sceneMap[sceneId];
            return (
              <div key={sceneId}>
                <div className="mb-3 flex items-center gap-2">
                  <span className="text-[10px] tracking-widest uppercase text-[#D4AF37] font-semibold">Scene {scene?.scene_number || '?'}</span>
                  <span className="text-xs text-[#71717A]">{scene?.purpose?.split('—')[0] || ''}</span>
                </div>
                <DndContext sensors={sensors} collisionDetection={closestCenter} onDragEnd={handleDragEnd}>
                  <SortableContext items={sceneShots.map(s => s.id)} strategy={verticalListSortingStrategy}>
                    <div className="space-y-3">{sceneShots.map((shot, idx) => <SortableShot key={shot.id} shot={shot} idx={idx} projectId={projectId} onShotUpdate={handleShotUpdate} />)}</div>
                  </SortableContext>
                </DndContext>
              </div>
            );
          })}
        </div>
      </ScrollArea>
    </div>
  );
}
