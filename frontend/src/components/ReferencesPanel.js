import { useState, useEffect } from "react";
import { Users, MapPin, Plus, Trash2, Save, ChevronDown, ChevronUp, Shirt, Upload, Image as ImageIcon } from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { toast } from "sonner";
import api from "@/lib/api";

const BACKEND = process.env.REACT_APP_BACKEND_URL;

export default function ReferencesPanel({ projectId }) {
  const [characters, setCharacters] = useState([]);
  const [environments, setEnvironments] = useState([]);
  const [showAddChar, setShowAddChar] = useState(false);
  const [showAddEnv, setShowAddEnv] = useState(false);
  const [newChar, setNewChar] = useState({ name: "", role: "", description: "", appearance: "", wardrobe: "", age_range: "" });
  const [newEnv, setNewEnv] = useState({ name: "", description: "", time_of_day: "", mood: "", architecture: "", visual_details: "" });

  useEffect(() => {
    api.listCharacters(projectId).then(setCharacters).catch(() => {});
    api.listEnvironments(projectId).then(setEnvironments).catch(() => {});
  }, [projectId]);

  const addChar = async () => {
    if (!newChar.name.trim()) { toast.error("Name required"); return; }
    try { const c = await api.createCharacter(projectId, newChar); setCharacters(p => [...p, c]); setNewChar({ name: "", role: "", description: "", appearance: "", wardrobe: "", age_range: "" }); setShowAddChar(false); toast.success(`"${c.name}" added`); } catch { toast.error("Failed"); }
  };

  const addEnv = async () => {
    if (!newEnv.name.trim()) { toast.error("Name required"); return; }
    try { const e = await api.createEnvironment(projectId, newEnv); setEnvironments(p => [...p, e]); setNewEnv({ name: "", description: "", time_of_day: "", mood: "", architecture: "", visual_details: "" }); setShowAddEnv(false); toast.success(`"${e.name}" added`); } catch { toast.error("Failed"); }
  };

  return (
    <div className="h-full flex flex-col">
      <div className="px-6 py-4 border-b border-[#27272A] shrink-0">
        <p className="panel-overline mb-1">References</p>
        <h2 className="font-heading text-2xl text-[#F3F4F6]">Cast & Locations</h2>
        <p className="text-sm text-[#71717A] mt-1">Upload reference images. Add look variants per scene. All injected into prompts via inheritance.</p>
      </div>
      <Tabs defaultValue="characters" className="flex-1 flex flex-col overflow-hidden">
        <TabsList className="mx-6 mt-3 bg-[#1E1E24]">
          <TabsTrigger value="characters" data-testid="tab-characters" className="data-[state=active]:bg-[#D4AF37]/10 data-[state=active]:text-[#D4AF37]"><Users className="w-3.5 h-3.5 mr-1.5" />Characters ({characters.length})</TabsTrigger>
          <TabsTrigger value="environments" data-testid="tab-environments" className="data-[state=active]:bg-[#D4AF37]/10 data-[state=active]:text-[#D4AF37]"><MapPin className="w-3.5 h-3.5 mr-1.5" />Locations ({environments.length})</TabsTrigger>
        </TabsList>

        <TabsContent value="characters" className="flex-1 overflow-hidden mt-0">
          <ScrollArea className="h-full"><div className="p-6 space-y-3">
            {characters.map((c, i) => <CharCard key={c.id} char={c} idx={i} pid={projectId} onUpdate={u => setCharacters(p => p.map(x => x.id === u.id ? u : x))} onDelete={() => { api.deleteCharacter(projectId, c.id); setCharacters(p => p.filter(x => x.id !== c.id)); toast.success("Removed"); }} />)}
            {showAddChar ? (
              <div className="bg-[#141417] border border-[#D4AF37]/30 rounded-lg p-4 space-y-2 animate-fade-up" data-testid="add-char-form">
                <Input value={newChar.name} onChange={e => setNewChar(p => ({...p, name: e.target.value}))} placeholder="Character name" className="bg-[#1E1E24] border-[#27272A] text-[#F3F4F6] text-sm" data-testid="char-name-input" />
                <div className="grid grid-cols-2 gap-2">
                  <Input value={newChar.role} onChange={e => setNewChar(p => ({...p, role: e.target.value}))} placeholder="Role" className="bg-[#1E1E24] border-[#27272A] text-[#F3F4F6] text-sm" />
                  <Input value={newChar.age_range} onChange={e => setNewChar(p => ({...p, age_range: e.target.value}))} placeholder="Age" className="bg-[#1E1E24] border-[#27272A] text-[#F3F4F6] text-sm" />
                </div>
                <Textarea value={newChar.appearance} onChange={e => setNewChar(p => ({...p, appearance: e.target.value}))} placeholder="Physical appearance" className="bg-[#1E1E24] border-[#27272A] text-[#F3F4F6] text-sm min-h-[50px]" rows={2} />
                <Input value={newChar.wardrobe} onChange={e => setNewChar(p => ({...p, wardrobe: e.target.value}))} placeholder="Default wardrobe" className="bg-[#1E1E24] border-[#27272A] text-[#F3F4F6] text-sm" />
                <div className="flex gap-2">
                  <Button onClick={addChar} size="sm" className="bg-[#D4AF37] text-[#0A0A0C] hover:bg-[#F1C40F] text-xs" data-testid="save-char-btn"><Save className="w-3 h-3 mr-1" />Save</Button>
                  <Button onClick={() => setShowAddChar(false)} variant="outline" size="sm" className="border-[#27272A] text-[#A1A1AA] text-xs">Cancel</Button>
                </div>
              </div>
            ) : (
              <Button onClick={() => setShowAddChar(true)} variant="outline" className="w-full border-dashed border-[#27272A] text-[#71717A] hover:text-[#D4AF37] hover:border-[#D4AF37]/30" data-testid="add-char-btn"><Plus className="w-3.5 h-3.5 mr-1.5" />Add Character</Button>
            )}
          </div></ScrollArea>
        </TabsContent>

        <TabsContent value="environments" className="flex-1 overflow-hidden mt-0">
          <ScrollArea className="h-full"><div className="p-6 space-y-3">
            {environments.map((e, i) => <EnvCard key={e.id} env={e} idx={i} pid={projectId} onUpdate={u => setEnvironments(p => p.map(x => x.id === u.id ? u : x))} onDelete={() => { api.deleteEnvironment(projectId, e.id); setEnvironments(p => p.filter(x => x.id !== e.id)); toast.success("Removed"); }} />)}
            {showAddEnv ? (
              <div className="bg-[#141417] border border-[#C85A17]/30 rounded-lg p-4 space-y-2 animate-fade-up" data-testid="add-env-form">
                <Input value={newEnv.name} onChange={e => setNewEnv(p => ({...p, name: e.target.value}))} placeholder="Location name" className="bg-[#1E1E24] border-[#27272A] text-[#F3F4F6] text-sm" data-testid="env-name-input" />
                <Textarea value={newEnv.description} onChange={e => setNewEnv(p => ({...p, description: e.target.value}))} placeholder="Description" className="bg-[#1E1E24] border-[#27272A] text-[#F3F4F6] text-sm min-h-[50px]" rows={2} />
                <div className="grid grid-cols-2 gap-2">
                  <Input value={newEnv.time_of_day} onChange={e => setNewEnv(p => ({...p, time_of_day: e.target.value}))} placeholder="Time of day" className="bg-[#1E1E24] border-[#27272A] text-[#F3F4F6] text-sm" />
                  <Input value={newEnv.mood} onChange={e => setNewEnv(p => ({...p, mood: e.target.value}))} placeholder="Mood" className="bg-[#1E1E24] border-[#27272A] text-[#F3F4F6] text-sm" />
                </div>
                <Input value={newEnv.architecture} onChange={e => setNewEnv(p => ({...p, architecture: e.target.value}))} placeholder="Architecture" className="bg-[#1E1E24] border-[#27272A] text-[#F3F4F6] text-sm" />
                <div className="flex gap-2">
                  <Button onClick={addEnv} size="sm" className="bg-[#C85A17] text-white hover:bg-[#D4AF37] text-xs" data-testid="save-env-btn"><Save className="w-3 h-3 mr-1" />Save</Button>
                  <Button onClick={() => setShowAddEnv(false)} variant="outline" size="sm" className="border-[#27272A] text-[#A1A1AA] text-xs">Cancel</Button>
                </div>
              </div>
            ) : (
              <Button onClick={() => setShowAddEnv(true)} variant="outline" className="w-full border-dashed border-[#27272A] text-[#71717A] hover:text-[#C85A17] hover:border-[#C85A17]/30" data-testid="add-env-btn"><Plus className="w-3.5 h-3.5 mr-1.5" />Add Location</Button>
            )}
          </div></ScrollArea>
        </TabsContent>
      </Tabs>
    </div>
  );
}

function CharCard({ char, idx, pid, onUpdate, onDelete }) {
  const [expanded, setExpanded] = useState(false);
  const [showVariant, setShowVariant] = useState(false);
  const [nv, setNv] = useState({ label: "", wardrobe: "", appearance_notes: "" });

  const handleUpload = async (e) => {
    const file = e.target.files?.[0]; if (!file) return;
    try { const u = await api.uploadCharacterRef(pid, char.id, file); onUpdate(u); toast.success("Reference uploaded"); }
    catch { toast.error("Upload failed"); }
  };
  const addVariant = async () => {
    if (!nv.label.trim()) { toast.error("Label required"); return; }
    try { const u = await api.addLookVariant(pid, char.id, nv); onUpdate(u); setNv({ label: "", wardrobe: "", appearance_notes: "" }); setShowVariant(false); toast.success("Variant added"); }
    catch { toast.error("Failed"); }
  };
  const delVariant = async (vid) => {
    try { const u = await api.deleteLookVariant(pid, char.id, vid); onUpdate(u); toast.success("Removed"); } catch { toast.error("Failed"); }
  };

  const variants = char.look_variants || [];

  return (
    <div data-testid={`char-card-${idx}`} className="bg-[#141417] border border-[#27272A] rounded-lg overflow-hidden film-slate">
      <div className="p-4 flex gap-4">
        {/* Reference image */}
        <div className="shrink-0">
          {char.reference_image_url ? (
            <div className="relative group">
              <img src={`${BACKEND}${char.reference_image_url}`} alt={char.name} className="w-20 h-20 rounded-lg object-cover border border-[#27272A]" />
              <label className="absolute inset-0 flex items-center justify-center bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity rounded-lg cursor-pointer">
                <Upload className="w-4 h-4 text-white" />
                <input type="file" accept="image/*" className="hidden" onChange={handleUpload} />
              </label>
            </div>
          ) : (
            <label className="w-20 h-20 rounded-lg border-2 border-dashed border-[#27272A] flex flex-col items-center justify-center cursor-pointer hover:border-[#D4AF37]/40 transition-colors" data-testid={`upload-char-${idx}`}>
              <Upload className="w-4 h-4 text-[#3f3f46] mb-1" />
              <span className="text-[8px] text-[#3f3f46]">Upload</span>
              <input type="file" accept="image/*" className="hidden" onChange={handleUpload} />
            </label>
          )}
        </div>
        {/* Info */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between mb-1">
            <div className="flex items-center gap-2">
              <h4 className="text-sm font-semibold text-[#F3F4F6]">{char.name}</h4>
              <span className="text-[9px] bg-[#D4AF37]/10 text-[#D4AF37] px-1.5 py-0.5 rounded uppercase">{char.role}</span>
              {char.age_range && <span className="text-[10px] text-[#4a4a55]">{char.age_range}</span>}
              {variants.length > 0 && <span className="text-[9px] bg-[#8B5CF6]/10 text-[#8B5CF6] px-1.5 py-0.5 rounded">{variants.length} look{variants.length !== 1 ? 's' : ''}</span>}
            </div>
            <div className="flex gap-1">
              <button onClick={() => setExpanded(!expanded)} className="p-1 rounded hover:bg-[#27272E] text-[#4a4a55]">{expanded ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}</button>
              <button onClick={onDelete} className="p-1 rounded hover:bg-[#27272E] text-[#71717A] hover:text-[#EF4444]"><Trash2 className="w-3.5 h-3.5" /></button>
            </div>
          </div>
          {char.appearance && <p className="text-xs text-[#A1A1AA] truncate"><span className="text-[#4a4a55]">Look: </span>{char.appearance}</p>}
          {char.wardrobe && <p className="text-xs text-[#71717A] truncate"><span className="text-[#4a4a55]">Wardrobe: </span>{char.wardrobe}</p>}
        </div>
      </div>

      {expanded && (
        <div className="border-t border-[#1F1F24] bg-[#0C0C0E] p-4 space-y-3 animate-fade-up">
          {char.description && <p className="text-xs text-[#71717A]"><span className="text-[#4a4a55]">Background: </span>{char.description}</p>}
          <div className="flex items-center justify-between mb-1">
            <p className="text-[10px] tracking-widest uppercase text-[#71717A] font-semibold"><Shirt className="w-3 h-3 inline mr-1" />Look Variants</p>
            <Button onClick={() => setShowVariant(true)} variant="ghost" size="sm" className="text-[#D4AF37] hover:bg-[#D4AF37]/10 text-[10px] h-6 px-2" data-testid={`add-variant-${idx}`}><Plus className="w-3 h-3 mr-0.5" />Add Look</Button>
          </div>
          {variants.length === 0 && !showVariant && <p className="text-[10px] text-[#3f3f46]">No variants — uses default wardrobe</p>}
          {variants.map((v, vi) => (
            <div key={v.id} className="bg-[#141417] border border-[#1F1F24] rounded-lg p-3 flex items-start justify-between" data-testid={`variant-${idx}-${vi}`}>
              <div><span className="text-xs font-medium text-[#8B5CF6]">{v.label}</span>
                {v.wardrobe && <p className="text-[10px] text-[#A1A1AA] mt-0.5">{v.wardrobe}</p>}
                {v.appearance_notes && <p className="text-[10px] text-[#71717A]">{v.appearance_notes}</p>}
              </div>
              <button onClick={() => delVariant(v.id)} className="p-0.5 rounded hover:bg-[#27272E] text-[#3f3f46] hover:text-[#EF4444]"><Trash2 className="w-3 h-3" /></button>
            </div>
          ))}
          {showVariant && (
            <div className="bg-[#141417] border border-[#8B5CF6]/30 rounded-lg p-3 space-y-2 animate-fade-up">
              <Input value={nv.label} onChange={e => setNv(p => ({...p, label: e.target.value}))} placeholder="Label (Wedding, Night, Flashback)" className="bg-[#1E1E24] border-[#27272A] text-[#F3F4F6] text-xs h-8" data-testid={`variant-label-${idx}`} />
              <Input value={nv.wardrobe} onChange={e => setNv(p => ({...p, wardrobe: e.target.value}))} placeholder="Wardrobe for this look" className="bg-[#1E1E24] border-[#27272A] text-[#F3F4F6] text-xs h-8" />
              <Input value={nv.appearance_notes} onChange={e => setNv(p => ({...p, appearance_notes: e.target.value}))} placeholder="Appearance changes" className="bg-[#1E1E24] border-[#27272A] text-[#F3F4F6] text-xs h-8" />
              <div className="flex gap-2">
                <Button onClick={addVariant} size="sm" className="bg-[#8B5CF6] text-white hover:bg-[#7C3AED] text-[10px] h-7" data-testid={`save-variant-${idx}`}><Save className="w-3 h-3 mr-1" />Save</Button>
                <Button onClick={() => setShowVariant(false)} variant="ghost" size="sm" className="text-[#4a4a55] text-[10px] h-7">Cancel</Button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function EnvCard({ env, idx, pid, onUpdate, onDelete }) {
  const handleUpload = async (e) => {
    const file = e.target.files?.[0]; if (!file) return;
    try { const u = await api.uploadEnvironmentRef(pid, env.id, file); onUpdate(u); toast.success("Reference uploaded"); }
    catch { toast.error("Upload failed"); }
  };

  return (
    <div data-testid={`env-card-${idx}`} className="bg-[#141417] border border-[#27272A] rounded-lg overflow-hidden film-slate">
      <div className="p-4 flex gap-4">
        <div className="shrink-0">
          {env.reference_image_url ? (
            <div className="relative group">
              <img src={`${BACKEND}${env.reference_image_url}`} alt={env.name} className="w-24 h-16 rounded-lg object-cover border border-[#27272A]" />
              <label className="absolute inset-0 flex items-center justify-center bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity rounded-lg cursor-pointer">
                <Upload className="w-4 h-4 text-white" />
                <input type="file" accept="image/*" className="hidden" onChange={handleUpload} />
              </label>
            </div>
          ) : (
            <label className="w-24 h-16 rounded-lg border-2 border-dashed border-[#27272A] flex flex-col items-center justify-center cursor-pointer hover:border-[#C85A17]/40 transition-colors" data-testid={`upload-env-${idx}`}>
              <Upload className="w-4 h-4 text-[#3f3f46] mb-0.5" />
              <span className="text-[8px] text-[#3f3f46]">Upload</span>
              <input type="file" accept="image/*" className="hidden" onChange={handleUpload} />
            </label>
          )}
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-start justify-between mb-1">
            <div><h4 className="text-sm font-semibold text-[#F3F4F6]">{env.name}</h4>
              <div className="flex items-center gap-2 text-[10px]">
                {env.mood && <span className="text-[#C85A17]">{env.mood}</span>}
                {env.time_of_day && <span className="text-[#71717A]">{env.time_of_day}</span>}
              </div>
            </div>
            <button onClick={onDelete} className="p-1 rounded hover:bg-[#27272E] text-[#71717A] hover:text-[#EF4444]"><Trash2 className="w-3.5 h-3.5" /></button>
          </div>
          {env.description && <p className="text-xs text-[#A1A1AA] line-clamp-2">{env.description}</p>}
          {env.architecture && <p className="text-[10px] text-[#4a4a55] mt-0.5">{env.architecture}</p>}
        </div>
      </div>
    </div>
  );
}
