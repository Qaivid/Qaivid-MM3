import { useState } from "react";
import { Copy, Download, ArrowRight } from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";

export default function PromptPanel({ prompts, shots, scenes, onExport, onNext }) {
  const [selectedModel, setSelectedModel] = useState("all");

  if (!prompts || prompts.length === 0) {
    return (
      <div className="h-full flex items-center justify-center text-[#71717A]">
        <p>No prompts yet. Generate prompts from the Shots panel.</p>
      </div>
    );
  }

  const shotMap = {};
  shots.forEach(s => { shotMap[s.id] = s; });
  const sceneMap = {};
  scenes.forEach(s => { sceneMap[s.id] = s; });

  const filteredPrompts = selectedModel === "all"
    ? prompts
    : prompts.filter(p => p.model_target === selectedModel);

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text);
    toast.success("Copied to clipboard");
  };

  return (
    <div className="h-full flex flex-col">
      <div className="px-6 py-4 border-b border-[#27272A] flex items-center justify-between shrink-0">
        <div>
          <p className="panel-overline mb-1">Generation</p>
          <h2 className="font-heading text-2xl text-[#F3F4F6]">{prompts.length} Prompt{prompts.length !== 1 ? 's' : ''}</h2>
        </div>
        <div className="flex gap-2">
          <Button
            data-testid="export-all-btn"
            variant="outline"
            size="sm"
            onClick={() => onExport("json")}
            className="border-[#27272A] text-[#A1A1AA] hover:text-white hover:bg-[#27272E] text-xs"
          >
            <Download className="w-3 h-3 mr-1" />
            Export All
          </Button>
        </div>
      </div>

      <ScrollArea className="flex-1">
        <div className="p-6 space-y-4">
          {/* Continue banner */}
          <div className="bg-[#141417] border border-[#D4AF37]/20 rounded-xl p-4 flex items-center justify-between">
            <div>
              <p className="text-sm font-semibold text-[#F3F4F6]">Prompts ready</p>
              <p className="text-[11px] text-[#71717A] mt-0.5">Generate reference images, shot stills, and render your video clips.</p>
            </div>
            <Button
              data-testid="continue-to-production-btn"
              onClick={onNext}
              className="bg-[#D4AF37] text-[#0A0A0C] hover:bg-[#F1C40F] font-semibold text-xs shrink-0 ml-4"
            >
              Continue to Production
              <ArrowRight className="w-3.5 h-3.5 ml-1.5" />
            </Button>
          </div>
          {filteredPrompts.map((prompt, idx) => {
            const shot = shotMap[prompt.shot_id] || {};
            const scene = sceneMap[prompt.scene_id] || {};
            return (
              <div
                key={prompt.id}
                data-testid={`prompt-card-${idx}`}
                className="bg-[#141417] border border-[#27272A] rounded-lg p-4 animate-fade-up"
                style={{ animationDelay: `${idx * 40}ms` }}
              >
                {/* Header */}
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <span className="text-[10px] tracking-widest uppercase text-[#71717A] font-semibold">
                      Scene {scene.scene_number || '?'} / Shot {shot.shot_number || '?'}
                    </span>
                    <span className="text-[10px] bg-[#D4AF37]/10 text-[#D4AF37] px-1.5 py-0.5 rounded font-medium">
                      {prompt.model_target}
                    </span>
                    <span className="text-[10px] text-[#71717A]">{prompt.aspect_ratio} &middot; {prompt.duration}s</span>
                  </div>
                  <button
                    data-testid={`copy-prompt-${idx}`}
                    onClick={() => copyToClipboard(prompt.positive_prompt)}
                    className="p-1.5 rounded hover:bg-[#27272E] text-[#71717A] hover:text-[#F3F4F6] transition-colors"
                  >
                    <Copy className="w-3.5 h-3.5" />
                  </button>
                </div>

                {/* Positive Prompt */}
                <div className="bg-[#1E1E24] rounded p-3 mb-2">
                  <p className="text-[10px] tracking-widest uppercase text-[#10B981] font-semibold mb-1">Positive</p>
                  <p className="text-xs text-[#F3F4F6] leading-relaxed font-mono">{prompt.positive_prompt}</p>
                </div>

                {/* Negative Prompt */}
                {prompt.negative_prompt && (
                  <div className="bg-[#1E1E24] rounded p-3 mb-2">
                    <p className="text-[10px] tracking-widest uppercase text-[#EF4444] font-semibold mb-1">Negative</p>
                    <p className="text-xs text-[#A1A1AA] leading-relaxed font-mono">{prompt.negative_prompt}</p>
                  </div>
                )}

                {/* Style Injection */}
                {prompt.style_injection && (
                  <div className="mt-2 text-xs">
                    <span className="text-[#71717A]">Style: </span>
                    <span className="text-[#D4AF37]">{prompt.style_injection}</span>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </ScrollArea>
    </div>
  );
}
