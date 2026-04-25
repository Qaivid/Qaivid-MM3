import crypto from "crypto";
import { URL as NodeURL } from "url";
import sharp from "sharp";
import { uploadToR2, downloadFromR2, getPresignedDownloadUrl, streamFromR2 } from "../r2";

const SSRF_BLOCKED_PATTERNS = [
  /^localhost$/i,
  /^127\./,
  /^0\.0\.0\.0$/,
  /^::1$/,
  /^10\./,
  /^172\.(1[6-9]|2\d|3[01])\./,
  /^192\.168\./,
  /^169\.254\./,
  /^100\.(6[4-9]|[7-9]\d|1[01]\d|12[0-7])\./,
  /^fd/i,
  /^fe80:/i,
  /^metadata\.google\.internal$/i,
  /^169\.254\.169\.254$/,
];

function assertSafeExternalUrl(rawUrl: string): void {
  let parsed: NodeURL;
  try {
    parsed = new NodeURL(rawUrl);
  } catch {
    throw new Error(`Invalid URL: ${rawUrl.substring(0, 80)}`);
  }
  if (parsed.protocol !== "https:") {
    throw new Error(`Only HTTPS URLs are allowed for external image sources (got: ${parsed.protocol})`);
  }
  const hostname = parsed.hostname.toLowerCase();
  for (const pattern of SSRF_BLOCKED_PATTERNS) {
    if (pattern.test(hostname)) {
      throw new Error(`Blocked URL: hostname "${hostname}" is not allowed for external image fetch`);
    }
  }
}

const RUNWARE_API_URL = "https://api.runware.ai/v1";

export interface OutpaintModel {
  id: string;
  name: string;
  description: string;
  costPerImage: string;
}

export const OUTPAINT_MODELS: OutpaintModel[] = [
  {
    id: "runware:102@1",
    name: "FLUX.1 Fill Dev",
    description: "Open-source inpainting/outpainting model. High-quality scene extension with coherent content generation.",
    costPerImage: "~$0.0064",
  },
];

async function runwarePost(tasks: any[]): Promise<any[]> {
  const apiKey = process.env.RUNWARE_API_KEY;
  if (!apiKey) throw new Error("RUNWARE_API_KEY not configured");

  const response = await fetch(RUNWARE_API_URL, {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(tasks),
  });

  const result = await response.json() as any;

  if (!response.ok || result.errors?.length > 0) {
    const errMsg = result.errors?.[0]?.message || `HTTP ${response.status}`;
    throw new Error(`Runware API error: ${errMsg}`);
  }

  return result.data || [];
}

async function uploadImageToRunware(imageSource: string): Promise<string> {
  const taskUUID = crypto.randomUUID();
  console.log(`[Outpaint] Uploading source image to Runware...`);

  const results = await runwarePost([{
    taskType: "imageUpload",
    taskUUID,
    image: imageSource,
  }]);

  const result = results.find((r: any) => r.taskUUID === taskUUID) || results[0];
  if (!result?.imageUUID) throw new Error("Runware image upload failed — no imageUUID returned");
  console.log(`[Outpaint] Source uploaded: ${result.imageUUID}`);
  return result.imageUUID as string;
}

// Canonical output resolutions — the definitive "already correct" check
const TARGET_RESOLUTIONS: Record<string, { width: number; height: number }> = {
  "16:9": { width: 1920, height: 1080 },
  "9:16": { width: 1080, height: 1920 },
};

interface OutpaintExtension {
  finalWidth: number;
  finalHeight: number;
  left: number;
  right: number;
  top: number;
  bottom: number;
}

/**
 * Returns outpaint extension amounts to bring the image to the correct aspect ratio for Runware,
 * or null if no extension is needed (image already has correct ratio or is wider/taller — direct
 * resize to canonical target is sufficient in that case).
 * Throws only when the image is already at the exact canonical output resolution.
 */
function resolveOutpaintDimensions(
  originalWidth: number,
  originalHeight: number,
  targetAspectRatio: string,
): OutpaintExtension | null {
  const roundTo64 = (n: number) => Math.round(n / 64) * 64;

  const targetRes = TARGET_RESOLUTIONS[targetAspectRatio];
  if (!targetRes) {
    throw new Error(
      `Unsupported target aspect ratio: ${targetAspectRatio}. Only 16:9 and 9:16 are supported for outpainting.`,
    );
  }

  // Already at canonical output resolution — no work needed at all
  if (originalWidth === targetRes.width && originalHeight === targetRes.height) {
    throw new Error(
      `This image is already ${targetRes.width}×${targetRes.height} (${targetAspectRatio}). No outpainting needed.`,
    );
  }

  if (targetAspectRatio === "16:9") {
    const targetWidth = roundTo64(originalHeight * (16 / 9));
    const totalExtension = targetWidth - originalWidth;
    if (totalExtension <= 0) {
      // Image is already wide enough for its height — no AI extension needed, just resize to canonical
      return null;
    }
    const half = Math.floor(totalExtension / 2);
    return {
      finalWidth: targetWidth,
      finalHeight: originalHeight,
      left: half,
      right: totalExtension - half,
      top: 0,
      bottom: 0,
    };
  }

  // 9:16
  const targetHeight = roundTo64(originalWidth * (16 / 9));
  const totalExtension = targetHeight - originalHeight;
  if (totalExtension <= 0) {
    // Image is already tall enough for its width — no AI extension needed, just resize to canonical
    return null;
  }
  const half = Math.floor(totalExtension / 2);
  return {
    finalWidth: originalWidth,
    finalHeight: targetHeight,
    left: 0,
    right: 0,
    top: half,
    bottom: totalExtension - half,
  };
}

export interface OutpaintRequest {
  imageUrl: string;
  prompt: string;
  modelId: string;
  targetAspectRatio: string;
  projectId: number;
  shotId: number;
}

/** Read actual image dimensions from an R2 object without downloading the full file.
 *  For PNGs: reads only the 24-byte IHDR header via a range request (very fast).
 *  For any other format (JPEG, WebP, etc.): falls back to downloading the full file and using sharp.
 */
async function getR2ImageDimensions(r2Key: string): Promise<{ width: number; height: number }> {
  // Read the first 24 bytes — enough for a PNG signature (8 bytes) + IHDR chunk (16 bytes)
  const { body } = await streamFromR2(r2Key, "bytes=0-23");
  const headerBytes = await new Promise<Buffer>((resolve, reject) => {
    const chunks: Buffer[] = [];
    body.on("data", (chunk: Buffer) => chunks.push(chunk));
    body.on("end", () => resolve(Buffer.concat(chunks)));
    body.on("error", reject);
  });

  // PNG magic: 89 50 4E 47 0D 0A 1A 0A
  const isPng =
    headerBytes.length >= 8 &&
    headerBytes[0] === 0x89 &&
    headerBytes[1] === 0x50 &&
    headerBytes[2] === 0x4e &&
    headerBytes[3] === 0x47;

  if (isPng && headerBytes.length >= 24) {
    // PNG IHDR: bytes 16-19 = width (big-endian uint32), bytes 20-23 = height
    const width = headerBytes.readUInt32BE(16);
    const height = headerBytes.readUInt32BE(20);
    if (width > 0 && height > 0) {
      console.log(`[Outpaint] PNG header dimensions: ${width}×${height}`);
      return { width, height };
    }
  }

  // Non-PNG or unexpected header — download full image and detect via sharp
  console.log(`[Outpaint] Non-PNG R2 image detected, downloading for dimension detection`);
  const buf = await downloadFromR2(r2Key);
  const meta = await sharp(buf).metadata();
  if (!meta.width || !meta.height) throw new Error("Could not determine image dimensions from R2 asset");
  console.log(`[Outpaint] Sharp-detected dimensions: ${meta.width}×${meta.height}`);
  return { width: meta.width, height: meta.height };
}

export async function outpaintImage(request: OutpaintRequest): Promise<{ imageUrl: string }> {
  const { imageUrl, prompt, modelId, targetAspectRatio, projectId, shotId } = request;

  const model = OUTPAINT_MODELS.find((m) => m.id === modelId);
  if (!model) {
    throw new Error(
      `Unknown outpaint model: "${modelId}". Valid models: ${OUTPAINT_MODELS.map((m) => m.id).join(", ")}`,
    );
  }

  console.log(`[Outpaint] Shot ${shotId}: model=${modelId}, target=${targetAspectRatio}, imageUrl=${imageUrl.substring(0, 80)}`);

  let originalWidth: number;
  let originalHeight: number;
  let r2Key: string | null = null;
  let sourceBuffer: Buffer | null = null; // only populated for external URLs

  if (imageUrl.startsWith("/api/r2/")) {
    r2Key = decodeURIComponent(imageUrl.replace("/api/r2/", "").split("?")[0]);
    const detected = await getR2ImageDimensions(r2Key);
    originalWidth = detected.width;
    originalHeight = detected.height;
    console.log(`[Outpaint] Actual R2 image dimensions: ${originalWidth}×${originalHeight}`);
  } else if (imageUrl.startsWith("https://")) {
    assertSafeExternalUrl(imageUrl);
    const res = await fetch(imageUrl);
    if (!res.ok) throw new Error(`Failed to download source image: HTTP ${res.status}`);
    sourceBuffer = Buffer.from(await res.arrayBuffer());
    const metadata = await sharp(sourceBuffer).metadata();
    if (!metadata.width || !metadata.height) throw new Error("Could not determine image dimensions");
    originalWidth = metadata.width;
    originalHeight = metadata.height;
    console.log(`[Outpaint] Downloaded external image: ${originalWidth}×${originalHeight}`);
  } else {
    throw new Error(
      `Source image URL must be an internal /api/r2/ path or a public HTTPS URL. Got: ${imageUrl.substring(0, 80)}`,
    );
  }

  // Compute outpaint extension amounts — null means no AI fill needed, just resize to canonical
  const extension = resolveOutpaintDimensions(originalWidth, originalHeight, targetAspectRatio);

  const targetRes = TARGET_RESOLUTIONS[targetAspectRatio]!;

  if (!extension) {
    // Image ratio is already wide/tall enough — no Runware call needed. Resize directly to canonical.
    console.log(`[Outpaint] No AI extension needed — resizing directly to ${targetRes.width}×${targetRes.height}`);
    const imgBuf = r2Key ? await downloadFromR2(r2Key) : sourceBuffer!;
    const resized = await sharp(imgBuf)
      .resize(targetRes.width, targetRes.height, { fit: "fill" })
      .webp({ quality: 90 })
      .toBuffer();
    const outKey = `projects/${projectId}/shots/${shotId}/outpaint_${Date.now()}.webp`;
    const outUrl = await uploadToR2(outKey, resized, "image/webp");
    console.log(`[Outpaint] Resize-only result saved to R2: ${outUrl}`);
    return { imageUrl: outUrl };
  }

  console.log(
    `[Outpaint] Extension: left=${extension.left} right=${extension.right} top=${extension.top} bottom=${extension.bottom} → ${extension.finalWidth}×${extension.finalHeight}`,
  );

  // Build Runware source: presigned URL for R2 (Runware fetches directly), base64 for external
  let runwareImageSource: string;
  if (r2Key) {
    runwareImageSource = await getPresignedDownloadUrl(r2Key, 300); // 5-min presigned URL
    console.log(`[Outpaint] Using presigned R2 URL (no server download)`);
  } else {
    const mimeType = (await sharp(sourceBuffer!).metadata()).format === "jpeg" ? "image/jpeg" : "image/png";
    runwareImageSource = `data:${mimeType};base64,${sourceBuffer!.toString("base64")}`;
  }

  const imageUUID = await uploadImageToRunware(runwareImageSource);

  // Submit outpaint task
  const taskUUID = crypto.randomUUID();
  const task: any = {
    taskType: "imageInference",
    taskUUID,
    model: modelId,
    positivePrompt: prompt,
    seedImage: imageUUID,
    outpaint: {
      left: extension.left,
      right: extension.right,
      top: extension.top,
      bottom: extension.bottom,
    },
    width: extension.finalWidth,
    height: extension.finalHeight,
    steps: 30,
    numberResults: 1,
  };

  console.log(`[Outpaint] Submitting imageInference task ${taskUUID}`);
  const submitResults = await runwarePost([task]);

  function extractImageUrl(r: any): string | undefined {
    return r?.imageURL ?? r?.imageUrl ?? r?.url ?? undefined;
  }

  // Check if result came back synchronously (some Runware image models do this)
  let resultImageUrl: string | undefined;
  const syncResult = submitResults.find((r: any) => r.taskUUID === taskUUID) || submitResults[0];
  if (syncResult) {
    resultImageUrl = extractImageUrl(syncResult);
    if (resultImageUrl) {
      console.log(`[Outpaint] Got synchronous result from Runware`);
    }
  }

  // If not returned synchronously, poll with getResponse (same pattern as video)
  if (!resultImageUrl) {
    console.log(`[Outpaint] Polling for result with taskUUID=${taskUUID}...`);
    const POLL_INTERVAL_MS = 2000;
    const POLL_TIMEOUT_MS = 120_000;
    const startTime = Date.now();

    while (Date.now() - startTime < POLL_TIMEOUT_MS) {
      await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS));

      const pollResults = await runwarePost([{
        taskType: "getResponse",
        taskUUID,
      }]);

      if (pollResults && pollResults.length > 0) {
        const pollResult = pollResults.find((r: any) => r.taskUUID === taskUUID) || pollResults[0];
        if (pollResult) {
          resultImageUrl = extractImageUrl(pollResult);
          if (resultImageUrl) {
            console.log(`[Outpaint] Poll succeeded after ${Math.round((Date.now() - startTime) / 1000)}s`);
            break;
          }
          const status = pollResult.status?.toLowerCase();
          if (status === "failed" || status === "error") {
            throw new Error(`Runware outpaint failed: ${pollResult.error || pollResult.message || "Unknown error"}`);
          }
        }
      }
      console.log(`[Outpaint] Still processing... (${Math.round((Date.now() - startTime) / 1000)}s elapsed)`);
    }

    if (!resultImageUrl) {
      throw new Error(`Runware outpaint timed out after ${POLL_TIMEOUT_MS / 1000}s — no image URL received`);
    }
  }

  console.log(`[Outpaint] Result URL from Runware: ${resultImageUrl.substring(0, 80)}`);

  // Download result and store in R2
  const resultRes = await fetch(resultImageUrl);
  if (!resultRes.ok) throw new Error(`Failed to download outpainted image: HTTP ${resultRes.status}`);
  let resultBuffer = Buffer.from(await resultRes.arrayBuffer());

  // Scale to exact target resolution so video gen receives a true 16:9 / 9:16 frame
  // and convert to JPEG for fast R2 upload (~300KB vs ~4.5MB for PNG)
  const targetResolution =
    targetAspectRatio === "16:9" ? { width: 1920, height: 1080 } :
    targetAspectRatio === "9:16" ? { width: 1080, height: 1920 } : null;

  if (targetResolution) {
    const outpaintMeta = await sharp(resultBuffer).metadata();
    const needsScale =
      outpaintMeta.width !== targetResolution.width ||
      outpaintMeta.height !== targetResolution.height;
    if (needsScale) {
      console.log(
        `[Outpaint] Scaling ${outpaintMeta.width}×${outpaintMeta.height} → ${targetResolution.width}×${targetResolution.height} for exact ${targetAspectRatio}`,
      );
    }
    resultBuffer = await sharp(resultBuffer)
      .resize(targetResolution.width, targetResolution.height, { fit: "fill" })
      .webp({ quality: 90 })
      .toBuffer();
    console.log(`[Outpaint] WebP output: ${(resultBuffer.length / 1024).toFixed(0)}KB`);
  }

  const outpaintKey = `projects/${projectId}/shots/${shotId}/outpaint_${Date.now()}.webp`;
  const r2Url = await uploadToR2(outpaintKey, resultBuffer, "image/webp");

  console.log(`[Outpaint] Saved to R2: ${r2Url}`);
  return { imageUrl: r2Url };
}
