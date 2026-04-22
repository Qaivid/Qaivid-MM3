"""Email sending via Resend (resend.com).

Required environment secrets:
  RESEND_API_KEY     — Resend API key (starts with re_)
  RESEND_FROM_EMAIL  — verified sender address, e.g. noreply@yourdomain.com
  RESEND_FROM_NAME   — (optional) display name, default "Qaivid MetaMind"
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def _api_key() -> str:
    return os.getenv("RESEND_API_KEY", "")


def _from_addr() -> str:
    name  = os.getenv("RESEND_FROM_NAME", "Qaivid MetaMind")
    email = os.getenv("RESEND_FROM_EMAIL", "")
    if not email:
        return f"{name} <noreply@example.com>"
    return f"{name} <{email}>"


def send_verification_email(to_email: str, verify_url: str, site_name: str = "Qaivid MetaMind") -> bool:
    """Send an email-verification link. Returns True on success."""
    key = _api_key()
    if not key:
        logger.warning(
            "Email sending skipped (RESEND_API_KEY not set). Verification URL: %s", verify_url
        )
        return False

    brand_first = site_name.split()[0]
    brand_rest  = " ".join(site_name.split()[1:]) or "MetaMind 3.1"

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  body{{margin:0;padding:0;background:#0a0a0f;font-family:'Helvetica Neue',Arial,sans-serif;color:#e0e0e0;}}
  .wrap{{max-width:560px;margin:40px auto;background:#111118;border-radius:16px;overflow:hidden;border:1px solid rgba(255,255,255,0.07);}}
  .hero{{background:linear-gradient(135deg,#0a0a0f 0%,#111118 100%);padding:36px 40px 28px;border-bottom:1px solid rgba(255,255,255,0.07);}}
  .brand{{font-size:20px;font-weight:800;color:#fff;letter-spacing:-0.03em;}}
  .brand span{{color:#d4ff3a;}}
  .body{{padding:32px 40px;}}
  h2{{margin:0 0 12px;font-size:22px;font-weight:700;color:#fff;}}
  p{{margin:0 0 20px;font-size:15px;line-height:1.6;color:#b0b8c4;}}
  .btn{{display:inline-block;background:#d4ff3a;color:#0a0a0f;font-weight:700;font-size:15px;padding:14px 32px;border-radius:10px;text-decoration:none;letter-spacing:-0.01em;}}
  .url-fallback{{margin-top:20px;padding:12px 16px;background:#0a0a0f;border-radius:8px;font-size:12px;color:#666;word-break:break-all;}}
  .footer{{padding:20px 40px;font-size:12px;color:#444;border-top:1px solid rgba(255,255,255,0.05);}}
</style>
</head>
<body>
<div class="wrap">
  <div class="hero">
    <div class="brand">{brand_first}<span> {brand_rest}</span></div>
  </div>
  <div class="body">
    <h2>Confirm your email address</h2>
    <p>Thanks for signing up! Click the button below to verify your email and activate your account.</p>
    <p><a href="{verify_url}" class="btn">Verify my email</a></p>
    <p style="margin-top:24px;font-size:13px;">This link expires in <strong>24 hours</strong>. If you did not create an account, you can safely ignore this email.</p>
    <div class="url-fallback">
      <strong>Can't click the button?</strong> Copy this link:<br>{verify_url}
    </div>
  </div>
  <div class="footer">&copy; {site_name}. You received this because someone signed up with this address.</div>
</div>
</body>
</html>"""

    try:
        import resend
        resend.api_key = key
        resend.Emails.send({
            "from": _from_addr(),
            "to": [to_email],
            "subject": f"Verify your {site_name} account",
            "html": html,
        })
        logger.info("Verification email sent to %s", to_email)
        return True
    except Exception as exc:
        logger.error("Failed to send verification email to %s: %s", to_email, exc)
        return False


def send_password_reset_email(to_email: str, reset_url: str, site_name: str = "Qaivid MetaMind") -> bool:
    """Send a password-reset link. Returns True on success."""
    key = _api_key()
    if not key:
        logger.warning(
            "Email sending skipped (RESEND_API_KEY not set). Reset URL: %s", reset_url
        )
        return False

    brand_first = site_name.split()[0]
    brand_rest  = " ".join(site_name.split()[1:]) or "MetaMind 3.1"

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  body{{margin:0;padding:0;background:#0a0a0f;font-family:'Helvetica Neue',Arial,sans-serif;color:#e0e0e0;}}
  .wrap{{max-width:560px;margin:40px auto;background:#111118;border-radius:16px;overflow:hidden;border:1px solid rgba(255,255,255,0.07);}}
  .hero{{background:linear-gradient(135deg,#0a0a0f 0%,#111118 100%);padding:36px 40px 28px;border-bottom:1px solid rgba(255,255,255,0.07);}}
  .brand{{font-size:20px;font-weight:800;color:#fff;letter-spacing:-0.03em;}}
  .brand span{{color:#d4ff3a;}}
  .body{{padding:32px 40px;}}
  h2{{margin:0 0 12px;font-size:22px;font-weight:700;color:#fff;}}
  p{{margin:0 0 20px;font-size:15px;line-height:1.6;color:#b0b8c4;}}
  .btn{{display:inline-block;background:#d4ff3a;color:#0a0a0f;font-weight:700;font-size:15px;padding:14px 32px;border-radius:10px;text-decoration:none;letter-spacing:-0.01em;}}
  .url-fallback{{margin-top:20px;padding:12px 16px;background:#0a0a0f;border-radius:8px;font-size:12px;color:#666;word-break:break-all;}}
  .footer{{padding:20px 40px;font-size:12px;color:#444;border-top:1px solid rgba(255,255,255,0.05);}}
</style>
</head>
<body>
<div class="wrap">
  <div class="hero">
    <div class="brand">{brand_first}<span> {brand_rest}</span></div>
  </div>
  <div class="body">
    <h2>Reset your password</h2>
    <p>We received a request to reset the password for your account. Click the button below to choose a new password.</p>
    <p><a href="{reset_url}" class="btn">Reset my password</a></p>
    <p style="margin-top:24px;font-size:13px;">This link expires in <strong>1 hour</strong>. If you did not request a password reset, you can safely ignore this email — your password will not change.</p>
    <div class="url-fallback">
      <strong>Can't click the button?</strong> Copy this link:<br>{reset_url}
    </div>
  </div>
  <div class="footer">&copy; {site_name}. You received this because a password reset was requested for this address.</div>
</div>
</body>
</html>"""

    try:
        import resend
        resend.api_key = key
        resend.Emails.send({
            "from": _from_addr(),
            "to": [to_email],
            "subject": f"Reset your {site_name} password",
            "html": html,
        })
        logger.info("Password reset email sent to %s", to_email)
        return True
    except Exception as exc:
        logger.error("Failed to send password reset email to %s: %s", to_email, exc)
        return False


def send_welcome_email(to_email: str, site_name: str = "Qaivid MetaMind") -> bool:
    """Send a welcome email after successful verification (fire-and-forget)."""
    key = _api_key()
    if not key:
        return False

    brand_first = site_name.split()[0]
    brand_rest  = " ".join(site_name.split()[1:]) or "MetaMind 3.1"

    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<style>
  body{{margin:0;padding:0;background:#0a0a0f;font-family:'Helvetica Neue',Arial,sans-serif;color:#e0e0e0;}}
  .wrap{{max-width:560px;margin:40px auto;background:#111118;border-radius:16px;overflow:hidden;border:1px solid rgba(255,255,255,0.07);padding:40px;}}
  .brand{{font-size:20px;font-weight:800;color:#fff;margin-bottom:24px;}}
  .brand span{{color:#d4ff3a;}}
  h2{{margin:0 0 12px;font-size:22px;font-weight:700;color:#fff;}}
  p{{margin:0 0 16px;font-size:15px;line-height:1.6;color:#b0b8c4;}}
  .btn{{display:inline-block;background:#d4ff3a;color:#0a0a0f;font-weight:700;font-size:15px;padding:14px 32px;border-radius:10px;text-decoration:none;}}
</style></head><body>
<div class="wrap">
  <div class="brand">{brand_first}<span> {brand_rest}</span></div>
  <h2>You're all set!</h2>
  <p>Your account is active. Start creating cinematic music videos — paste your lyrics and let MetaMind do the rest.</p>
  <p><a href="#" class="btn">Go to studio &rarr;</a></p>
</div></body></html>"""

    try:
        import resend
        resend.api_key = key
        resend.Emails.send({
            "from": _from_addr(),
            "to": [to_email],
            "subject": f"Welcome to {site_name}!",
            "html": html,
        })
        return True
    except Exception as exc:
        logger.error("Failed to send welcome email to %s: %s", to_email, exc)
        return False
