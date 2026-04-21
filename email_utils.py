"""Email sending via SendGrid.

Required environment secrets:
  SENDGRID_API_KEY   — SendGrid API key (starts with SG.)
  SENDGRID_FROM_EMAIL — verified sender address in SendGrid
  SENDGRID_FROM_NAME  — (optional) display name, default "Qaivid MetaMind"

If SENDGRID_API_KEY is not set the functions log a warning and return False
so the rest of the app can continue to function during development.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def _sg_client():
    api_key = os.getenv("SENDGRID_API_KEY", "")
    if not api_key:
        return None
    try:
        import sendgrid
        return sendgrid.SendGridAPIClient(api_key=api_key)
    except ImportError:
        logger.error("sendgrid package not installed — run: pip install sendgrid")
        return None


def _from_addr() -> dict:
    email = os.getenv("SENDGRID_FROM_EMAIL", os.getenv("ADMIN_EMAIL", "noreply@example.com"))
    name  = os.getenv("SENDGRID_FROM_NAME", "Qaivid MetaMind")
    return {"email": email, "name": name}


def send_verification_email(to_email: str, verify_url: str, site_name: str = "Qaivid MetaMind") -> bool:
    """Send an email-verification link. Returns True on success."""
    sg = _sg_client()
    if not sg:
        logger.warning(
            "Email sending skipped (SENDGRID_API_KEY not set). "
            "Verification URL: %s", verify_url
        )
        return False

    try:
        from sendgrid.helpers.mail import Mail, To
        html = f"""
<!DOCTYPE html>
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
  .btn{{display:inline-block;background:#d4ff3a;color:#0a0a0f;font-weight:700;font-size:15px;
        padding:14px 32px;border-radius:10px;text-decoration:none;letter-spacing:-0.01em;}}
  .url-fallback{{margin-top:20px;padding:12px 16px;background:#0a0a0f;border-radius:8px;
                 font-size:12px;color:#666;word-break:break-all;}}
  .footer{{padding:20px 40px;font-size:12px;color:#444;border-top:1px solid rgba(255,255,255,0.05);}}
</style>
</head>
<body>
<div class="wrap">
  <div class="hero">
    <div class="brand">{site_name.split()[0]}<span> {' '.join(site_name.split()[1:]) or 'MetaMind 3.1'}</span></div>
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
  <div class="footer">&copy; {site_name}. You're receiving this because someone signed up with this email address.</div>
</div>
</body>
</html>
"""
        message = Mail(
            from_email=(_from_addr()["email"], _from_addr()["name"]),
            to_emails=to_email,
            subject=f"Verify your {site_name} account",
            html_content=html,
        )
        sg.send(message)
        logger.info("Verification email sent to %s", to_email)
        return True
    except Exception as exc:
        logger.error("Failed to send verification email to %s: %s", to_email, exc)
        return False


def send_welcome_email(to_email: str, site_name: str = "Qaivid MetaMind") -> bool:
    """Send a welcome email after successful verification (fire-and-forget)."""
    sg = _sg_client()
    if not sg:
        return False
    try:
        from sendgrid.helpers.mail import Mail
        html = f"""
<!DOCTYPE html><html><head><meta charset="UTF-8">
<style>
  body{{margin:0;padding:0;background:#0a0a0f;font-family:'Helvetica Neue',Arial,sans-serif;color:#e0e0e0;}}
  .wrap{{max-width:560px;margin:40px auto;background:#111118;border-radius:16px;overflow:hidden;
         border:1px solid rgba(255,255,255,0.07);padding:40px;}}
  .brand{{font-size:20px;font-weight:800;color:#fff;margin-bottom:24px;}}
  .brand span{{color:#d4ff3a;}}
  h2{{margin:0 0 12px;font-size:22px;font-weight:700;color:#fff;}}
  p{{margin:0 0 16px;font-size:15px;line-height:1.6;color:#b0b8c4;}}
  .btn{{display:inline-block;background:#d4ff3a;color:#0a0a0f;font-weight:700;font-size:15px;
        padding:14px 32px;border-radius:10px;text-decoration:none;}}
</style></head><body>
<div class="wrap">
  <div class="brand">{site_name.split()[0]}<span> {' '.join(site_name.split()[1:]) or 'MetaMind 3.1'}</span></div>
  <h2>You're all set!</h2>
  <p>Your account is active. Start creating cinematic music videos — just paste your lyrics and let MetaMind do the rest.</p>
  <p><a href="https://qaivid.com" class="btn">Go to studio &rarr;</a></p>
</div></body></html>
"""
        message = Mail(
            from_email=(_from_addr()["email"], _from_addr()["name"]),
            to_emails=to_email,
            subject=f"Welcome to {site_name}!",
            html_content=html,
        )
        sg.send(message)
        return True
    except Exception as exc:
        logger.error("Failed to send welcome email to %s: %s", to_email, exc)
        return False
