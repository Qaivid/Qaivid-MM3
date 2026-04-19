import { useState } from "react";
import { DndContext, closestCenter, PointerSensor, useSensor, useSensors } from "@dnd-kit/core";
import { SortableContext, verticalListSortingStrategy, useSortable } from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";
import { Clapperboard, Layers, MapPin, Clock, Thermometer, GripVertical, Pencil, Check, X } from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { toast } from "sonner";
import api from "@/lib/api";

const TEMPORAL_COLORS = {
  present: "bg-[#10B981]/20 text-[#10B981]",
  memory: "bg-[#3B82F6]/20 text-[#3B82F6]",
  symbolic: "bg-[#D4AF37]/20 text-[#D4AF37]",
};

function SortableScene({ scene, idx, projectId, onSceneUpdate }) {
  const { attributes, listeners, setNodeRef, transform, transition, isDragging } = useSortable({ id: scene.id });
  const style = { transform: CSS.Transform.toString(transform), transition, opacity: isDragging ? 0.5 : 1, zIndex: isDragging ? 50 : 'auto' };
  const [editing, setEditing] = useState(false);
  const [edits, setEdits] = useState({});

  const startEdit = () => {
    setEdits({ location: scene.location || "", time_of_day: scene.time_of_day || "", emotional_temperature: scene.emotional_temperature || "", temporal_status: scene.temporal_status || "present", purpose: scene.purpose || "" });
    setEditing(true);
  };

  const saveEdit = async () => {
    try {
      const updated = await api.updateScene(projectId, scene.id, edits);
      onSceneUpdate(updated);
      setEditing(false);
      toast.success("Scene updated");
    } catch { toast.error("Update failed"); }
  };

  return (
    <div ref={setNodeRef} style={style} data-testid={`scene-card-${idx}`}
      className="film-slate bg-[#141417] border border-[#27272A] rounded-lg p-5 animate-fade-up" {...attributes}>
      <div className="flex items-start gap-3 mb-3">
        <button {...listeners} className="mt-1 p-1 rounded hover:bg-[#27272E] text-[#3f3f46] hover:text-[#D4AF37] cursor-grab active:cursor-grabbing shrink-0" data-testid={`drag-scene-${idx}`}>
          <GripVertical className="w-4 h-4" />
        </button>
        <div className="flex-1 min-w-0">
          <div className="flex items-start justify-between">
            <div className="flex-1 min-w-0">
              <span className="text-[10px] tracking-widest uppercase text-[#71717A] font-semibold">Scene {scene.scene_number}</span>
              {editing ? (
                <Input value={edits.purpose} onChange={e => setEdits(p => ({...p, purpose: e.target.value}))}
                  className="bg-[#1E1E24] border-[#27272A] text-[#F3F4F6] text-sm h-8 mt-1" />
              ) : (
                <h3 className="text-sm font-semibold text-[#F3F4F6] mt-0.5">{scene.purpose}</h3>
              )}
            </div>
            <div className="flex items-center gap-1.5 ml-2 shrink-0">
              {editing ? (
                <Select value={edits.temporal_status} onValueChange={v => setEdits(p => ({...p, temporal_status: v}))}>
                  <SelectTrigger className="bg-[#1E1E24] border-[#27272A] text-[#F3F4F6] text-[10px] h-7 w-24"><SelectValue /></SelectTrigger>
                  <SelectContent className="bg-[#1E1E24] border-[#27272A]">
                    {["present","memory","symbolic"].map(t => <SelectItem key={t} value={t} className="text-[#F3F4F6] text-xs">{t}</SelectItem>)}
                  </SelectContent>
                </Select>
              ) : (
                <span className={`px-2 py-0.5 rounded text-[10px] uppercase font-semibold ${TEMPORAL_COLORS[scene.temporal_status] || TEMPORAL_COLORS.present}`}>{scene.temporal_status}</span>
              )}
              {editing ? (
                <>
                  <button onClick={saveEdit} className="p-1 rounded hover:bg-[#10B981]/20 text-[#10B981]" data-testid={`save-scene-${idx}`}><Check className="w-3.5 h-3.5" /></button>
                  <button onClick={() => setEditing(false)} className="p-1 rounded hover:bg-[#EF4444]/20 text-[#EF4444]"><X className="w-3.5 h-3.5" /></button>
                </>
              ) : (
                <button onClick={startEdit} className="p-1 rounded hover:bg-[#27272E] text-[#3f3f46] hover:text-[#D4AF37]" data-testid={`edit-scene-${idx}`}><Pencil className="w-3.5 h-3.5" /></button>
              )}
            </div>
          </div>
        </div>
      </div>

      {editing ? (
        <div className="grid grid-cols-3 gap-2 mb-3 ml-10">
          <Input value={edits.location} onChange={e => setEdits(p => ({...p, location: e.target.value}))} placeholder="Location" className="bg-[#1E1E24] border-[#27272A] text-[#F3F4F6] text-xs h-7" />
          <Input value={edits.time_of_day} onChange={e => setEdits(p => ({...p, time_of_day: e.target.value}))} placeholder="Time of day" className="bg-[#1E1E24] border-[#27272A] text-[#F3F4F6] text-xs h-7" />
          <Input value={edits.emotional_temperature} onChange={e => setEdits(p => ({...p, emotional_temperature: e.target.value}))} placeholder="Emotion" className="bg-[#1E1E24] border-[#27272A] text-[#F3F4F6] text-xs h-7" />
        </div>
      ) : (
        <div className="grid grid-cols-3 gap-3 mb-3 text-xs ml-10">
          <div className="flex items-center gap-1.5 text-[#A1A1AA]"><MapPin className="w-3 h-3 text-[#71717A]" /><span className="truncate">{scene.location || "unspecified"}</span></div>
          <div className="flex items-center gap-1.5 text-[#A1A1AA]"><Clock className="w-3 h-3 text-[#71717A]" /><span>{scene.time_of_day || "unspecified"}</span></div>
          <div className="flex items-center gap-1.5 text-[#A1A1AA]"><Thermometer className="w-3 h-3 text-[#71717A]" /><span className="truncate">{scene.emotional_temperature || "neutral"}</span></div>
        </div>
      )}

      <p className="text-xs text-[#71717A] mb-2 ml-10"><span className="text-[#A1A1AA] font-medium">Function: </span>{scene.story_function}</p>
      {scene.lyric_text && <div className="bg-[#1E1E24] rounded p-3 ml-10"><p className="font-heading text-sm text-[#A1A1AA] italic leading-relaxed whitespace-pre-line">{scene.lyric_text}</p></div>}
      {scene.objects_of_significance?.length > 0 && <div className="flex flex-wrap gap-1 mt-3 ml-10">{scene.objects_of_significance.map((obj, i) => <span key={i} className="text-[10px] bg-[#D4AF37]/10 text-[#D4AF37] px-2 py-0.5 rounded">{obj}</span>)}</div>}
      {scene.visual_risk_notes?.length > 0 && <div className="mt-3 ml-10 space-y-1">{scene.visual_risk_notes.map((r, i) => <p key={i} className="text-[10px] text-[#F59E0B]">{r}</p>)}</div>}
    </div>
  );
}

export default function ScenePanel({ scenes, setScenes, projectId, onBuildShots, buildingShots, onBuildScenes, buildingScenes }) {
  const sensors = useSensors(useSensor(PointerSensor, { activationConstraint: { distance: 8 } }));

  const handleDragEnd = async (event) => {
    const { active, over } = event;
    if (!over || active.id === over.id) return;
    const oldIdx = scenes.findIndex(s => s.id === active.id);
    const newIdx = scenes.findIndex(s => s.id === over.id);
    if (oldIdx === -1 || newIdx === -1) return;
    const reordered = [...scenes];
    const [moved] = reordered.splice(oldIdx, 1);
    reordered.splice(newIdx, 0, moved);
    const renumbered = reordered.map((s, i) => ({ ...s, scene_number: i + 1 }));
    setScenes(renumbered);
    try { await api.reorderScenes(projectId, renumbered.map(s => s.id)); toast.success("Reordered"); }
    catch { toast.error("Reorder failed"); setScenes(scenes); }
  };

  const handleSceneUpdate = (updated) => {
    setScenes(prev => prev.map(s => s.id === updated.id ? { ...s, ...updated } : s));
  };

  if (!scenes?.length) {
    return (
      <div className="h-full flex flex-col">
        <div className="px-6 py-4 border-b border-[#27272A] shrink-0">
          <p className="panel-overline mb-1">Storyboard</p>
          <h2 className="font-heading text-2xl text-[#F3F4F6]">Storyboard</h2>
        </div>
        <div className="flex-1 flex items-center justify-center p-8">
          <div className="max-w-sm text-center">
            <div className="w-16 h-16 rounded-2xl bg-[#8B5CF6]/10 border border-[#8B5CF6]/15 flex items-center justify-center mx-auto mb-5">
              <Layers className="w-8 h-8 text-[#8B5CF6]" />
            </div>
            <h3 className="text-lg font-semibold text-[#F3F4F6] mb-2">Build Your Storyboard</h3>
            <p className="text-sm text-[#71717A] mb-6">Scenes are structured from your Creative Brief and context intelligence — lyric-synced, arc-driven, ready for shot design.</p>
            <Button
              data-testid="build-scenes-btn"
              onClick={onBuildScenes}
              disabled={buildingScenes}
              className="bg-[#8B5CF6] text-white hover:bg-[#7C3AED] font-semibold w-full"
            >
              <Clapperboard className="w-4 h-4 mr-2" />
              {buildingScenes ? "Building Storyboard…" : "Build Storyboard"}
            </Button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col">
      <div className="px-6 py-4 border-b border-[#27272A] flex items-center justify-between shrink-0">
        <div>
          <p className="panel-overline mb-1">Storyboard</p>
          <h2 className="font-heading text-2xl text-[#F3F4F6]">{scenes.length} Scene{scenes.length !== 1 ? 's' : ''}</h2>
          <p className="text-[10px] text-[#4a4a55] mt-0.5">Drag to reorder. Click pencil to edit in place.</p>
        </div>
        <Button data-testid="build-shots-btn" onClick={onBuildShots} disabled={buildingShots}
          className="bg-[#D4AF37] text-[#0A0A0C] hover:bg-[#F1C40F] font-semibold text-xs">
          <Clapperboard className="w-3.5 h-3.5 mr-1.5" />{buildingShots ? "Building..." : "Build Shots"}
        </Button>
      </div>
      <ScrollArea className="flex-1">
        <div className="p-6 space-y-4">
          <DndContext sensors={sensors} collisionDetection={closestCenter} onDragEnd={handleDragEnd}>
            <SortableContext items={scenes.map(s => s.id)} strategy={verticalListSortingStrategy}>
              {scenes.map((scene, idx) => <SortableScene key={scene.id} scene={scene} idx={idx} projectId={projectId} onSceneUpdate={handleSceneUpdate} />)}
            </SortableContext>
          </DndContext>
        </div>
      </ScrollArea>
    </div>
  );
}
