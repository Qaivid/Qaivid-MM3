import axios from "axios";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const http = axios.create({ baseURL: API, withCredentials: true });

function formatError(detail) {
  if (detail == null) return "Something went wrong";
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) return detail.map(e => e?.msg || JSON.stringify(e)).join(" ");
  if (detail?.msg) return detail.msg;
  return String(detail);
}

const api = {
  // Auth
  login: (email, password) => http.post("/auth/login", { email, password }).then(r => r.data),
  register: (email, password, name) => http.post("/auth/register", { email, password, name }).then(r => r.data),
  logout: () => http.post("/auth/logout").then(r => r.data),
  getMe: () => http.get("/auth/me").then(r => r.data),
  refreshToken: () => http.post("/auth/refresh").then(r => r.data),

  // Projects
  listProjects: () => http.get("/projects").then(r => r.data),
  createProject: (data) => http.post("/projects", data).then(r => r.data),
  getProject: (id) => http.get(`/projects/${id}`).then(r => r.data),
  updateProject: (id, data) => http.put(`/projects/${id}`, data).then(r => r.data),
  deleteProject: (id) => http.delete(`/projects/${id}`).then(r => r.data),

  // Source Input
  addInput: (projectId, data) => http.post(`/projects/${projectId}/input`, data).then(r => r.data),
  getInput: (projectId) => http.get(`/projects/${projectId}/input`).then(r => r.data),

  // Interpretation
  interpret: (projectId) => http.post(`/projects/${projectId}/interpret`).then(r => r.data),
  getContext: (projectId) => http.get(`/projects/${projectId}/context`).then(r => r.data),

  // Overrides
  addOverride: (projectId, data) => http.post(`/projects/${projectId}/overrides`, data).then(r => r.data),
  getOverrides: (projectId) => http.get(`/projects/${projectId}/overrides`).then(r => r.data),
  deleteOverride: (projectId, overrideId) => http.delete(`/projects/${projectId}/overrides/${overrideId}`).then(r => r.data),

  // Scenes
  buildScenes: (projectId) => http.post(`/projects/${projectId}/scenes/build`).then(r => r.data),
  getScenes: (projectId) => http.get(`/projects/${projectId}/scenes`).then(r => r.data),
  updateScene: (projectId, sceneId, data) => http.put(`/projects/${projectId}/scenes/${sceneId}`, data).then(r => r.data),
  reorderScenes: (projectId, sceneIds) => http.put(`/projects/${projectId}/scenes/reorder`, { scene_ids: sceneIds }).then(r => r.data),

  // Shots
  buildShots: (projectId) => http.post(`/projects/${projectId}/shots/build`).then(r => r.data),
  getShots: (projectId) => http.get(`/projects/${projectId}/shots`).then(r => r.data),
  updateShot: (projectId, shotId, data) => http.put(`/projects/${projectId}/shots/${shotId}`, data).then(r => r.data),
  reorderShots: (projectId, shotIds) => http.put(`/projects/${projectId}/shots/reorder`, { shot_ids: shotIds }).then(r => r.data),

  // Prompts
  buildPrompts: (projectId, modelTarget = "generic") =>
    http.post(`/projects/${projectId}/prompts/build?model_target=${modelTarget}`).then(r => r.data),
  getPrompts: (projectId) => http.get(`/projects/${projectId}/prompts`).then(r => r.data),

  // Characters
  listCharacters: (projectId) => http.get(`/projects/${projectId}/characters`).then(r => r.data),
  createCharacter: (projectId, data) => http.post(`/projects/${projectId}/characters`, data).then(r => r.data),
  updateCharacter: (projectId, charId, data) => http.put(`/projects/${projectId}/characters/${charId}`, data).then(r => r.data),
  uploadCharacterRef: (projectId, charId, file) => {
    const form = new FormData(); form.append("file", file);
    return http.post(`/projects/${projectId}/characters/${charId}/upload-reference`, form, { headers: { "Content-Type": "multipart/form-data" } }).then(r => r.data);
  },
  deleteCharacter: (projectId, charId) => http.delete(`/projects/${projectId}/characters/${charId}`).then(r => r.data),
  addLookVariant: (projectId, charId, data) => http.post(`/projects/${projectId}/characters/${charId}/variants`, data).then(r => r.data),
  deleteLookVariant: (projectId, charId, variantId) => http.delete(`/projects/${projectId}/characters/${charId}/variants/${variantId}`).then(r => r.data),

  // Environments
  listEnvironments: (projectId) => http.get(`/projects/${projectId}/environments`).then(r => r.data),
  createEnvironment: (projectId, data) => http.post(`/projects/${projectId}/environments`, data).then(r => r.data),
  updateEnvironment: (projectId, envId, data) => http.put(`/projects/${projectId}/environments/${envId}`, data).then(r => r.data),
  uploadEnvironmentRef: (projectId, envId, file) => {
    const form = new FormData(); form.append("file", file);
    return http.post(`/projects/${projectId}/environments/${envId}/upload-reference`, form, { headers: { "Content-Type": "multipart/form-data" } }).then(r => r.data);
  },
  deleteEnvironment: (projectId, envId) => http.delete(`/projects/${projectId}/environments/${envId}`).then(r => r.data),

  // Continuity
  getContinuity: (projectId) => http.get(`/projects/${projectId}/continuity`).then(r => r.data),

  // Models
  listModels: () => http.get("/models").then(r => r.data),

  // Vibe Presets
  listVibePresets: () => http.get("/vibe-presets").then(r => r.data),
  getVibePreset: (id) => http.get(`/vibe-presets/${id}`).then(r => r.data),

  // Creative Brief
  generateBrief: (projectId, vibePreset = "") =>
    http.post(`/projects/${projectId}/brief?vibe_preset=${vibePreset}`).then(r => r.data),
  getBrief: (projectId) => http.get(`/projects/${projectId}/brief`).then(r => r.data),

  // Production Pipeline
  buildReferencePrompts: (projectId) => http.post(`/projects/${projectId}/reference-prompts`).then(r => r.data),
  getReferencePrompts: (projectId) => http.get(`/projects/${projectId}/reference-prompts`).then(r => r.data),
  buildStillPrompts: (projectId) => http.post(`/projects/${projectId}/still-prompts`).then(r => r.data),
  getStillPrompts: (projectId) => http.get(`/projects/${projectId}/still-prompts`).then(r => r.data),
  buildRenderPlan: (projectId, model = "wan_2_6") =>
    http.post(`/projects/${projectId}/render-plan?model=${model}`).then(r => r.data),
  getRenderPlan: (projectId) => http.get(`/projects/${projectId}/render-plan`).then(r => r.data),
  buildTimeline: (projectId) => http.post(`/projects/${projectId}/timeline`).then(r => r.data),
  getTimeline: (projectId) => http.get(`/projects/${projectId}/timeline`).then(r => r.data),
  getPipelineStatus: (projectId) => http.get(`/projects/${projectId}/pipeline`).then(r => r.data),

  // Audio
  uploadAudio: (projectId, file) => {
    const form = new FormData();
    form.append("file", file);
    return http.post(`/projects/${projectId}/audio`, form, { headers: { "Content-Type": "multipart/form-data" }, timeout: 120000 }).then(r => r.data);
  },
  getAudio: (projectId) => http.get(`/projects/${projectId}/audio`).then(r => r.data),

  // Real Generation
  generateReferences: (projectId) => http.post(`/projects/${projectId}/generate-references`, {}, { timeout: 300000 }).then(r => r.data),
  generateStills: (projectId) => http.post(`/projects/${projectId}/generate-stills`, {}, { timeout: 600000 }).then(r => r.data),
  renderVideos: (projectId) => http.post(`/projects/${projectId}/render-videos`, {}, { timeout: 300000 }).then(r => r.data),
  getRenderStatus: (projectId) => http.get(`/projects/${projectId}/render-status`).then(r => r.data),
  assembleVideo: (projectId) => http.post(`/projects/${projectId}/assemble`, {}, { timeout: 300000 }).then(r => r.data),
  getAssembly: (projectId) => http.get(`/projects/${projectId}/assembly`).then(r => r.data),

  // Validation
  validate: (projectId) => http.get(`/projects/${projectId}/validate`).then(r => r.data),

  // Export
  exportProject: (projectId, format) => http.get(`/projects/${projectId}/export/${format}`).then(r => r.data),

  // Culture Packs
  listCulturePacks: () => http.get("/culture-packs").then(r => r.data),
  getCulturePack: (packId) => http.get(`/culture-packs/${packId}`).then(r => r.data),


  // Admin
  getAdminStats: () => http.get("/admin/stats").then(r => r.data),
  getAdminUsers: () => http.get("/admin/users").then(r => r.data),
  getAdminUser: (id) => http.get(`/admin/users/${id}`).then(r => r.data),
  updateAdminUser: (id, data) => http.put(`/admin/users/${id}`, data).then(r => r.data),
  deleteAdminUser: (id) => http.delete(`/admin/users/${id}`).then(r => r.data),
  resetUserCredits: (id) => http.post(`/admin/users/${id}/reset-credits`).then(r => r.data),
  addUserCredits: (id, amount) => http.post(`/admin/users/${id}/add-credits`, { amount }).then(r => r.data),
  getAdminProjects: () => http.get("/admin/projects").then(r => r.data),
  getBillingConfig: () => http.get("/admin/billing-config").then(r => r.data),
  getAdminSettings: () => http.get("/admin/settings").then(r => r.data),
  updateAdminSettings: (data) => http.patch("/admin/settings", data).then(r => r.data),

  formatError,
};

export default api;
