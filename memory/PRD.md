# Qaivid 2.0 - PRD

## Product: Full SaaS AI Video Production Platform
Context Engine at the core. Multi-tenant with auth, admin, credits, and complete production pipeline.

## Architecture
- **Frontend**: React + TailwindCSS + Shadcn UI
- **Backend**: FastAPI + MongoDB (Motor async)
- **Auth**: JWT (httpOnly cookies) + bcrypt password hashing
- **Video**: AtlasCloud Wan 2.6 image-to-video
- **AI**: OpenAI GPT-4o + GPT Image 1 + Whisper | Gemini 2.5 Flash (audio transcription)

## Auth & User System
- Email + password registration/login with JWT
- httpOnly cookies (access_token 1hr, refresh_token 7 days)
- Brute force protection (5 attempts → 15 min lockout)
- Admin seeded on startup (admin@qaivid.com)
- Roles: user, admin
- Projects scoped to logged-in user

## Plans & Credits (from Qaivid 1.0)
| Plan | Credits | Price |
|------|---------|-------|
| Free | 0 | $0 |
| Starter | 1,500 | $14.99/mo |
| Pro | 4,000 | $39.99/mo |
| Studio | 20,000 | $199.99/mo |

### Operation Costs
- Interpretation: 10 credits
- Creative Brief: 10 credits  
- Scene Generation: 10 credits
- Image per shot: 6 credits
- Image per ref: 3 credits
- Transcription: 5 credits
- Video: 500 credits/min (normal), 100 credits/min (animatic)

## Admin Panel (/admin)
- Overview: user count, project count, plan distribution
- User management: view all, change plans, reset/add credits, delete
- Project overview: all projects across all users
- System API Keys: admin sets OpenAI, Gemini, AtlasCloud keys

## Full Pipeline (10 engines)
1. Source → 2. Context Intelligence → 3. Creative Brief → 4. Storyboard → 5. Shot Plan → 6. Prompts → 7. Reference Images → 8. Shot Stills → 9. Video Render → 10. Assembly

## API Keys (System-Wide, Admin-Managed)
| Key | Powers |
|-----|--------|
| OpenAI | GPT-4o, GPT Image 1, Whisper-1 |
| Gemini | Audio transcription (Punjabi/Hindi/Urdu native) |
| AtlasCloud | Video generation (Wan 2.6) |

## Completed Features
- [x] Auth system (JWT + bcrypt + brute force)
- [x] Admin panel (stats, user mgmt, project mgmt, API keys)
- [x] Credit system (plans, operation costs, ledger)
- [x] Full SaaS UI (Dashboard, Workspace, Auth, Admin)
- [x] 10-engine production pipeline
- [x] Dual-engine transcription (Gemini + Whisper)
- [x] AtlasCloud Wan 2.6 video generation
- [x] Secrets Manager (admin-managed)
- [x] 11 Vibe Presets, 7 Culture Packs, 12 Pacing Profiles
- [x] Drag-and-drop, inline editing, prompt inheritance
- [x] Character/environment profiles, reference images, timeline

## Backlog
### P1: Credit charging on operations (deduct credits when user runs pipeline steps)
### P1: Google OAuth social login
### P1: FFmpeg Video Assembly (stitch Wan 2.6 clips into final MP4)
### P2: True Continuity Tracking
### P3: Stripe billing integration, multi-language packs, collaboration
