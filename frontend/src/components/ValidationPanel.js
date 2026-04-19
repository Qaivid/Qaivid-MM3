import { X, AlertTriangle, AlertCircle, Info } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";

const SEVERITY_CONFIG = {
  error: { icon: AlertCircle, color: "text-[#EF4444]", bg: "bg-[#EF4444]/10", border: "border-[#EF4444]/20" },
  warning: { icon: AlertTriangle, color: "text-[#F59E0B]", bg: "bg-[#F59E0B]/10", border: "border-[#F59E0B]/20" },
  info: { icon: Info, color: "text-[#3B82F6]", bg: "bg-[#3B82F6]/10", border: "border-[#3B82F6]/20" },
};

export default function ValidationPanel({ validation, onClose }) {
  if (!validation) return null;

  return (
    <div className="h-full flex flex-col" data-testid="validation-panel">
      <div className="px-4 py-3 border-b border-[#27272A] flex items-center justify-between shrink-0">
        <div>
          <p className="panel-overline">Validation</p>
          <p className="text-xs text-[#A1A1AA] mt-0.5">
            {validation.errors} errors, {validation.warnings} warnings, {validation.infos} info
          </p>
        </div>
        <Button variant="ghost" size="sm" onClick={onClose} className="text-[#71717A] hover:text-white">
          <X className="w-4 h-4" />
        </Button>
      </div>

      <ScrollArea className="flex-1">
        <div className="p-4 space-y-2">
          {validation.total_issues === 0 ? (
            <div className="text-center py-8 text-[#10B981]">
              <p className="text-sm font-medium">All clear</p>
              <p className="text-xs text-[#71717A] mt-1">No validation issues found</p>
            </div>
          ) : (
            validation.issues?.map((issue, i) => {
              const config = SEVERITY_CONFIG[issue.severity] || SEVERITY_CONFIG.info;
              const Icon = config.icon;
              return (
                <div
                  key={i}
                  data-testid={`validation-issue-${i}`}
                  className={`${config.bg} border ${config.border} rounded-lg p-3`}
                >
                  <div className="flex items-start gap-2">
                    <Icon className={`w-3.5 h-3.5 ${config.color} mt-0.5 shrink-0`} />
                    <div className="min-w-0">
                      <div className="flex items-center gap-1.5 mb-0.5">
                        <span className={`text-[10px] uppercase font-semibold ${config.color}`}>
                          {issue.layer}
                        </span>
                      </div>
                      <p className="text-xs text-[#F3F4F6]">{issue.message}</p>
                      {issue.suggestion && (
                        <p className="text-[10px] text-[#71717A] mt-1">{issue.suggestion}</p>
                      )}
                    </div>
                  </div>
                </div>
              );
            })
          )}
        </div>
      </ScrollArea>
    </div>
  );
}
